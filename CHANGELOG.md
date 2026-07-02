# Changelog

All notable changes to Vincio are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [7.3.0] - 2026-07-02

**The packet compile receipt — prove *why* a packet was compiled, text-light.** Vincio already
treats the context packet as the governed boundary and traces the stages that produce it, but the
missing operational artifact — the one a reviewer wants when a run is surprising (a bad answer, stale
memory, a privacy-scope mismatch, budget trimming, or a replay divergence) — is a compact *receipt*
that explains the compile decision without exposing raw prompt or evidence text. `7.3` adds it. All
additive: new `vincio.context` symbols, one new `vincio trace receipt` command, and a new numbered
example, with **no existing symbol removed or changed**; `API_VERSION` stays `5.0`.

- **`CompileReceipt`** (new, in `vincio/context/receipt.py`, re-exported from `vincio.context`) — a
  fingerprint-heavy, text-light manifest of one context-packet compile: `packet_id` / `run_id` /
  `trace_id` (the pointer back to the trace), `compiler_version`, `policy_profile`, the
  `input_fingerprint`, a per-block **budget** summary, the **included** items with their citation
  locator, content hash, and selection scores (`score` / `relevance` / `authority` / `freshness`), the
  **excluded** items with their reason and `superseded_by`, the resolved and unresolved **conflicts**
  as winner / loser / rule, a **privacy** summary (scope, PII redaction count, scope-exclusion count,
  and the constant `omitted_raw_text: true`), and the **render** identity (provider, model,
  `context_ir_hash`, `rendered_packet_hash`). It carries ids, hashes, scores, and summaries — never raw
  text — so it is safe to attach to a PR, an incident note, or a bug report. Supporting models
  `ReceiptItem` / `ConflictSummary` / `BudgetSummary` / `PrivacySummary` / `RenderInfo` are re-exported
  too.
- **Deterministic, verifiable, diffable.** `receipt.receipt_hash` is a stable digest of the compile
  *decision* (excluding the per-run ids), so recompiling identical inputs yields the same hash and a
  changed source yields a different one; `receipt.diverges_from(baseline)` returns a structured
  divergence (items added/removed, score changes, budget delta, render change); `receipt.verify()`
  re-derives the receipt from its own bytes and checks its invariants.
- **Emitted and linked.** The packet and compiler carry the per-item selection scores and content
  hashes the receipt reads (`ContextPacket.compile_receipt(...)`, `CompiledContext.receipt(...)`), and
  every run builds a receipt and **links it from the trace** (set on the `prompt_render` span) and
  returns it on `result.metadata["compile_receipt"]`. Building the receipt is best-effort — it never
  breaks the run it observes.
- **CLI.** `vincio trace receipt <trace_id>` prints the receipt linked from a stored run's trace, with
  `--json` for the full artifact.
- **Docs, example, benchmarks.** A new concept, guide, and runnable example
  (`examples/17_compile_receipt.py`); a new `compile_receipt` VincioBench family and four published
  SLOs (deterministic, verifies-from-bytes, omits-raw-text, divergence-detected) held by
  at-least-as-strict budgets.

## [7.2.0] - 2026-07-01

**The benchmark platform — one system, three tracks.** The benchmark story had grown into four
loosely-related things (the public-benchmark plane, a competitive script, an uplift script, and the
internal VincioBench gate). `7.2` redesigns it into **one coherent platform with three tracks**, each
answering a distinct question and each supporting a **live** run and an offline **mockup**, under the
same provenance-tier honesty contract. All additive: new `vincio.evals.suite` symbols and a new
`vincio bench` command, with **no existing symbol removed or changed**; `API_VERSION` stays `5.0`.

The three tracks:

- **Track 1 — Model** (`vincio bench model`): how good is a *model* on the standard public benchmarks?
  The existing open evaluation plane — 29 benchmarks across 10 niches, Tier-S/R/L.
- **Track 2 — Uplift** (`vincio bench uplift`): how much does routing a model *through Vincio* change
  its scores? Each benchmark is scored **twice by the identical scorer** — the model's direct answer vs
  its Vincio-routed answer — and the per-benchmark delta is the measured uplift (grounding,
  prompt-injection containment, long-context needle recall, structured-output validity). Mockup replays
  two recorded arms deterministically; Live uses a real model for both arms.
- **Track 3 — Feature** (`vincio bench feature`): how good is a Vincio *feature* — memory, RAG,
  tokenization, output repair, prompt safety, tabular encoding, context assembly, chunking — vs the
  same feature in a competitor library? **Measured live on this machine** against the real library
  (`rank_bm25`, `tiktoken`, `json_repair`, `jinja2`, `pandas`, …); a missing competitor is reported
  *skipped*, never fabricated. The deterministic quality metric — not wall-clock — gates CI.

Every number carries a provenance tier — **L** Live (the real thing ran end to end: a live model, or a
real competitor library), **R** Recorded (a hash-pinned replay), **S** Static/Mockup (offline,
reproducible, gates CI) — and a lower tier can never print a higher tier's label. The internal
**VincioBench** gate is repositioned as exactly that (not a track) and now also CI-gates the
deterministic core of all three tracks via `families.bench_tracks.*`.

### Added

- **The track dimension** `BenchmarkTrack` (`model` / `uplift` / `feature`) and a unified
  **`vincio bench model|uplift|feature|list`** CLI, plus `python benchmarks/bench.py <track>`.
- **Track 2 — Uplift**: `UpliftSuite`, `UpliftBenchmark`, `UpliftResult`, `UpliftRun`, `UpliftRegistry`,
  `register_uplift_benchmark`, `render_uplift_report` (`vincio.evals.suite.uplift`). Four built-ins:
  `rag.grounded`, `safety.injection`, `long_context.recall`, `output.schema_valid`.
- **Track 3 — Feature**: `FeatureSuite`, `FeatureContest`, `Contender`, `FeatureMeasurement`,
  `FeatureRun`, `FeatureSuiteRun`, `FeatureRegistry`, `register_feature_contest`,
  `render_feature_report` (`vincio.evals.suite.feature_bench`). Eight built-in contests covering
  retrieval, tokenization, output repair, prompt safety, tabular encoding, context assembly, layered
  memory, and chunking — each vs a real competitor and/or a naive baseline.
- **CI gating for the new tracks**: a `bench_tracks` VincioBench family + `families.bench_tracks.*`
  budgets hold the deterministic, competitor-independent core of the uplift and feature tracks.
- **`benchmarks/bench.py`** — the folder driver for all three tracks (mirrors `vincio bench`).

### Changed

- **`benchmarks/manifest.json` and `PROVENANCE.md`** rewritten around the three tracks (each catalog
  folded live from its registry); the README benchmark section and the benchmark **assets are now all
  generated from data** by `benchmarks/render_assets.py` (`benchmark-platform`, `-plane`, `-uplift`,
  `-headtohead`) — no hand-transcribed numbers.
- **`benchmarks/competitive.py`** and **`quality_uplift.py`** are repositioned as the *extended* Track 3
  / Track 2 drivers; the canonical, tested, CI-gated logic now lives in the library.
- **Live re-evaluated against current state-of-the-art models** (OpenRouter, 2026-07-01, pinned in
  `benchmarks/reference/live_snapshot.json`): the uplift track — even `claude-opus-4.8` answers only
  ~13% of company-specific questions directly (abstaining, never hallucinating) but **97%** through
  Vincio's grounding, ~30× cheaper per correct answer; aggregate **13% → 95%** across four models. The
  feature track ran live vs the real libraries (BM25 ~12× faster, memory precision 1.0 vs 0.5, ~70%
  fewer tokens); the model track scored current models on real GSM8K/MMLU slices. Nothing fabricated.

### Fixed (post-release audit hardening)

A full adversarial audit of the benchmark system produced 32 verified findings, all resolved:

- **Honesty:** the uplift track no longer mislabels a fabricated mockup as `Recorded` — it refuses the
  tier (mirroring the model track's `resolve_tier`); and on every track a run's header tier is derived
  from its resolved per-result tiers (the feature suite folds *all* contests, not a looser
  any-competitor predicate), so it can never print a higher tier than any contest earned.
- **Robustness:** a feature contender that errors at runtime now degrades to a *skip* (never crashes the
  suite) and the contest drops to Static; feature contests use `run_sync` (safe inside an event loop)
  instead of bare `asyncio.run`; `benchmarks/eval_live.py` reports a clean error on a malformed dataset
  and always closes its run store; `check_budgets.py` handles an empty budget set; the chart rasterizer
  guards empty input.
- **Consistency & dead code:** the new tracks' content hash is named `determinism_digest` to match the
  model track; `UpliftBenchmark.higher_is_better` (per result *and* in the run-level overall) and
  `solver_mode` are honoured; the `MMLU-Pro` adapter renders its options on the live path like the other
  multiple-choice adapters; `UpliftRegistry` gains
  capability grouping; a dead helper, a stale competitor reference, and a vacuous loop removed; the
  `FeatureSuiteRun` id now covers every contest, not just the first.
- **Docs & tests:** stale four-plane references corrected to the three tracks; tests no longer pollute
  the process-wide registries, and new tests lock the honesty refusal, tier consistency,
  lower-is-better direction, runtime-crash resilience, and the report/error branches.

## [7.1.0] - 2026-07-01

**The evaluation plane, completed — provenance made legible, Recorded/Live made runnable.**
`7.0` shipped the open evaluation plane; `7.1` is the fit-and-finish that makes it *legible* and
*usable end to end*, and reconciles the rest of the benchmark story with it. All additive: no existing
symbol removed or changed, `API_VERSION` stays `5.0`.

The organizing idea is one honesty contract across **all four** benchmark planes, not just the new
one. Vincio measures itself with VincioBench (the internal mechanism suite), a competitive suite (vs
real libraries), an orchestrator-uplift suite (vs the raw model), and the open evaluation plane
(public benchmarks). Until now only the plane carried a structural provenance tier; the others stated
their honesty in prose. `7.1` gives every plane a machine-readable **evidence class** — Mechanism
(Tier-S), Comparative, Uplift, Recorded, Live — so a reader never has to guess whether a number is
`LIVE`, `STATIC/FABRICATED`, or a self-measurement.

The plane's **Recorded** and **Live** tiers were defined in `7.0` but not runnable out of the box: the
live path sent a bare question to the model without the multiple-choice options / RAG contexts /
long-context haystack its scorer looks for, so a live MMLU run would have scored ~0 by a harness bug,
not model weakness. `7.1` closes that: adapters render a **self-contained prompt** for the live path,
and a `benchmarks/eval_live.py` harness wires a `ContextApp` to any provider and runs the suite Live
against a current state-of-the-art model (`claude-opus-4-8`, `gpt-5.2`, `gemini-3-pro`, …) over a real
dataset — reported, never gated, and never fabricated (it refuses without a real key).

### Added

- **Benchmark provenance manifest & map.** `benchmarks/manifest.json` (generated from the live suite
  registry by `benchmarks/_manifest.py`) is the machine-readable source of truth for the four planes,
  the six evidence classes, and the 29-benchmark plane catalog; `benchmarks/PROVENANCE.md` is its
  human-readable "how real is each number?" map. `tests/test_benchmark_manifest.py` gates freshness and
  manifest↔registry consistency, and that the S/R/L codes line up with `ProvenanceTier`.
- **`BenchmarkAdapter.render_prompt(task)`** — the self-contained prompt a *live* solver sees. Overridden
  by the multiple-choice adapters (lettered options + an answer instruction), RAG faithfulness (the
  retrieved passages), and RULER (the long context), so a live model can actually answer and the
  extract-and-match scorer is fair. The offline replay/Tier-S path is unchanged.
- **`benchmarks/eval_live.py`** — run the plane Live against a SOTA model over a real dataset dir, with
  honest tier labelling and a hard refusal (no fabrication) when a key or dataset is missing.
- **`vincio eval suite list [--json]`** — the public-benchmark catalog by niche, each benchmark's primary
  metric and supported tiers, so the catalog is legible at a glance.
- **`benchmarks/render_assets.py`** — renders `assets/benchmark-plane.svg` deterministically from the
  manifest (single source of truth), in the repo's house style; embedded in the README.
- **`benchmarks/fixtures/README.md`** — states plainly that the fixtures are fabricated miniatures named
  after real datasets, not real dataset slices.

### Changed

- **README benchmarks section** reframed around the four planes and the S/R/L provenance tiers, with a
  new open-evaluation-plane subsection (catalog + the generated asset + a Live-SOTA example) and each
  existing subsection labelled with its evidence class.
- **Docs navigation** now surfaces the plane: `concepts/open-evaluation-plane.md` under Core concepts and
  `guides/run-benchmark-suite.md` under Evaluate and improve; `docs/reference/cli.md` documents
  `eval suite list`; `benchmarks/README.md` and `METHODOLOGY.md` cross-link the plane and the four-plane
  provenance contract.
- **Renamed** the VincioBench `breaking_2_0` family to `structural_guarantees` (a stable name for a
  permanent invariant set), across `vinciobench.py`, `budgets.json`, and `slos.json`.

### Fixed

- `builtin.py`'s catalog docstring said "13 … 16" adapters; the actual catalog is 15 niche + 14 agentic
  = 29 specs across ten niches.

## [7.0.0] - 2026-07-01

**The open evaluation plane.** Vincio already shipped an evaluation subsystem for *your* application
(golden datasets, 30+ metrics, calibrated judges) and a three-tier internal benchmark suite (VincioBench)
that proves the library's own mechanisms. This release adds the third thing: one coherent, pluggable
harness for running the **standard public model benchmarks** — MMLU, GPQA, GSM8K, HumanEval, IFEval,
TruthfulQA, RULER, and more — grouped by eleven niches behind the unchanged `BenchmarkAdapter` contract,
scored by reused metrics, and reported the same way for every model and every model *version*. It runs
in-process over your own store and never becomes a hosted leaderboard. Delivered **additively**: ten new
top-level entry points and the `app.benchmark_suite` verb, behind opt-in extras, with **no existing
symbol removed or changed**, so `API_VERSION` stays `5.0` and a clean upgrade needs zero source changes.
The 7.0 major marks the milestone, not a break.

The organizing idea is the **provenance tier** on every number, made an enforced contract rather than a
convention: **S — Static** (a bundled, *fabricated* fixture, reproducible/byte-identical, gates CI),
**R — Recorded** (a hash-pinned slice of the real dataset replayed against recorded outputs, reproducible
from the pin, gates CI), and **L — Live** (the full dataset against a live model, reported, never gated).
The engine computes the tier a run may claim from the dataset's provenance and whether the solver is live,
and **refuses** to let a lower tier print a higher tier's label — the project's honesty culture made
structural. Two differentiators fall out of the reuse: prompt injection reports *contained vs compromised*
(not merely attack-success), and every long-context benchmark runs twice, with and without the
`ContextGovernor`, so the uplift is measured, not assumed.

### Added

- **`vincio/evals/suite/` — the open evaluation plane (eight layers).** A new subpackage composing
  subsystems Vincio already ships: `BenchmarkSuite` (the deterministic, concurrent, resumable core engine
  with seeded sampling and checkpoint/resume), `BenchmarkRegistry` + `register_benchmark` (the niche-grouped
  catalog and the new `vincio.benchmarks` entry-point plugin group), `BenchmarkDataset` (content-addressed,
  hash-pinned task sets with an optional Hugging Face fetch), `ProvenanceTier` + `resolve_tier` (the tier
  contract), `SuiteRun`/`BenchmarkRun`/`ItemResult` (the tiered result models), `SuiteReport` (Markdown /
  HTML / JSON / CSV / PDF renderers that cite the exact scored items), `Leaderboard` (model and
  model-version ranking), the visualization builders (`leaderboard_chart`, `radar_chart`, `heatmap_chart`,
  `confusion_matrix_chart`, `trend_chart` — Vega-Lite by default), and `RunStore` (run history,
  `compare_runs`, `model_version_diff` over SQLite or Postgres).
- **Thirteen niche adapters for the standard public benchmarks**, joining the 16 re-homed agentic /
  text-to-query / data-analysis adapters under the unchanged `BenchmarkAdapter` contract: `MMLUAdapter`,
  `GPQAAdapter`, `ARCAdapter`, `HellaSwagAdapter`, `CEvalAdapter`, `CMMLUAdapter`, `TruthfulQAAdapter`,
  `GSM8KAdapter`, `MATHAdapter`, `HumanEvalAdapter`, `MBPPAdapter`, `IFEvalAdapter`, `PromptInjectionAdapter`,
  `RAGFaithfulnessAdapter`, and `RULERAdapter`, each scoring the benchmark's own verifiable criterion with a
  bundled Tier-S fabricated fixture and an official-export loader.
- **`app.benchmark_suite(...)`** — the thin `ContextApp` front that runs the plane over an app and audits
  the run on the hash-chained chain. `app.evaluate(...)` (golden-dataset eval → `EvalReport`) is unchanged.
- **Ten new top-level symbols** in `vincio.__all__`: `BenchmarkSuite`, `BenchmarkRegistry`, `BenchmarkSpec`,
  `register_benchmark`, `BenchmarkDataset`, `ProvenanceTier`, `SuiteRun`, `SuiteReport`, `Leaderboard`,
  `RunStore`.
- **CLI: `vincio eval suite {run,leaderboard,report,compare}`** — a sub-group beside the existing
  application-eval commands (`vincio eval run`, `vincio eval report`).
- **VincioBench `eval_suite` family + SLOs.** Proves the plane is honest the way `docs_conformance` proves
  the docs are connected: tier integrity, metric correctness, determinism (a Tier-S run is byte-identical),
  registry completeness, the measured long-context uplift, the report renderers, and a gate-bites companion
  (a mislabeled tier, a wrong metric, and a perturbed run are each caught), with four published SLOs.
- **Opt-in extras** keep the default dependency-free: `vincio[eval-datasets]` (Hugging Face `datasets`),
  `vincio[eval-viz]` (matplotlib PNG), `vincio[eval-pdf]` (reportlab), `vincio[eval-store]` (Postgres —
  SQLite is stdlib).
- **Two new error codes** — `EVAL_SUITE_ERROR` and `TIER_VIOLATION` (a lower tier refusing a higher tier's
  label) — each with a catalog entry and remediation.
- Concept page `docs/concepts/open-evaluation-plane.md`, guide `docs/guides/run-benchmark-suite.md`, and the
  runnable `examples/16_open_evaluation_plane.py`, wired into the docs graph.

## [6.6.0] - 2026-06-30

**Audit completion & standing guard — the hardening line (6.x) closes.** 6.0–6.5 acted on the core plane
(audited symbol-by-symbol) and swept the rest of the tree thematically. This release runs the same
symbol-by-symbol reachability rubric across *every* remaining subpackage — define the intended public surface
from `__init__`, then confirm every claim with a repo-wide reference check — and then locks the result in so the
interior quality the whole line buys cannot silently erode. `vincio._surface` already proves every public
`__all__` name *resolves*; it could not tell a live, *used* symbol from one that resolves yet is referenced
nowhere — the dead-but-resolvable surface 6.0 removed by a one-time *manual* reference check that was never
mechanized, so the exact debt was free to return. 6.6 mechanizes that check and gates the whole `hygiene`
family in CI as a first-class, named standing guard. Additive: the frozen top-level contract `vincio.__all__`
is unchanged (no public symbol is removed — the audit found none genuinely dead), so `API_VERSION` stays `5.0`,
the `vincio migrate 6.x` codemod table stays empty, and a clean upgrade needs zero source changes.

### Added

- **`vincio._reachability` — the public-symbol reachability guard.** It enumerates the whole public surface
  (the frozen top-level `vincio.__all__` plus every public subpackage's own `__all__`) and asks the audit's
  question of each symbol: is it *used* anywhere in the code corpus (`vincio/` + `tests/` + `examples/` +
  `benchmarks/`)? A use is a real load — an `ast.Name` load, an `obj.Symbol` attribute access, or an
  `import ... as alias` then a load of the alias; a bare re-export and an `__all__` entry are declarations, not
  uses. A symbol referenced nowhere is either *exercised by a new test* (the honest fix for a pure helper) or
  *declared, with its structural reason, in the frozen baseline* `docs/reference/reachability.txt` — an abstract
  base a user implements (`BASE`), an optional-dependency provider/backend (`OPTDEP`), or production wiring that
  binds a socket / Redis / webhook (`WIRING`). The gate holds the whole surface to that contract and provably
  bites on an injected dead-but-resolvable symbol and a stale baseline entry.
- **`tests/test_public_surface_reachability.py` — the audit's evidence.** The audit found 41 public symbols that
  resolved and were live capabilities yet were referenced nowhere in the corpus. Nineteen are pure,
  offline-runnable helpers with no structural excuse (the caching layers, `SchemaRegistry`, `LINT_RULES`,
  `SUPPORTED_EXTENSIONS`, the MCP OAuth helpers, `agui_sse`, `cache_hit_economics`, `analyze_ast_layout`,
  `extract_markdown_metadata`, `as_search_fn`, `map_compliance`, `residency_violation`, `CallableSparseEncoder`,
  `from_haystack_documents`, `MultiExporter`, `RoutingOptimizer`); each now has a focused behavioural test, which
  is the evidence the symbol is a real, working capability and the reason it is *not* in the baseline. The other
  twenty-two are structurally-unexercisable and declared in the baseline.
- **A dedicated `hygiene` CI job** runs all six interior-quality lints directly (`python -m vincio._surface`,
  `._error_contract`, `._observable_failure`, `._wire_or_retire`, `._assert_robustness`, `._reachability`) plus
  the eight gate test modules — a fast, named standing guard independent of the heavier benchmark job, so a
  hygiene regression shows a clear red check.
- **The `hygiene` VincioBench family gains the reachability half** (`reachability_conformant` /
  `reachability_clean` / `reachability_frozen` / `reachability_gate_detects_tamper`, plus the
  `hygiene_public_symbols_audited` / `hygiene_public_symbols_referenced` / `hygiene_unreferenced_public_symbols`
  / `hygiene_reachability_baseline` counts), with matching SLOs and at-least-as-strict budgets. The whole family
  — surface, error contract, observable failure, wire-or-retire, docstring parity, `-O` robustness, and now
  reachability — is gated in CI.

### Changed

- **`vincio._reachability` and `vincio._assert_robustness` join the `mypy --strict` set** (the regular
  `mypy vincio` overrides and the dedicated strict CI step), so the new guard ships under the same inline-typed
  contract as the rest of the hardening-line tooling.

## [6.5.0] - 2026-06-30

**`-O` robustness — the hardening line (6.x) continues.** Python strips every `assert` statement under
`python -O` (and `-OO`). A standing internal audit found a handful of `assert`s in shipped code that carried
real control-flow weight — narrowing a value the code then dereferenced, or checking a precondition a public
operation depended on — so in an optimized deployment they silently vanished, turning a caught invariant into
an opaque downstream `TypeError` / `AttributeError` far from its cause. This release replaces each load-bearing
`assert` with an explicit guard that raises the appropriate `VincioError`, so a `python -O` run fails loudly
and correctly, and keeps the genuine never-happens invariants as documented, marked `assert`s. Additive: the
frozen top-level contract `vincio.__all__` is unchanged, so `API_VERSION` stays `5.0`, the `vincio migrate 6.x`
codemod table stays empty, and a clean upgrade needs zero source changes. `except VincioError` (the documented
contract) was always the right catch and is unaffected — under `-O` it now *fires* where the stripped `assert`
used to be silent.

### Changed

- **The streaming and cascade response paths in `vincio.core.runtime` raise `ProviderResponseError`.** After a
  model tool loop completes, the runtime asserted a response had been produced (`assert response is not None`)
  in three places — the streaming executor, the cascade loop, and the streaming tool loop. Each is now an
  explicit guard: if the loop produced no response, it raises `ProviderResponseError` rather than relying on an
  `assert` that disappears under `-O` and yields a downstream `AttributeError` on `None`.
- **`vincio.context.compiler`'s resident-footprint fit raises `ContextCompileError`.** `_enforce_footprint`
  asserted the `max_resident_bytes` ceiling was configured before comparing the packet's estimate against it;
  it now raises `ContextCompileError` when invoked without a ceiling, so an `-O` deployment cannot silently
  compare against `None`.
- **`vincio.agents.graph`'s `CompiledGraph.ainvoke` raises `GraphError` on a missing terminal event.** The
  public `ainvoke` asserted the run ended with a terminal (`done` / `interrupt`) event before returning its
  payload; it now raises `GraphError` rather than returning `None` typed as a `GraphResult` under `-O`.
- **`vincio.mcp.transport`'s stdio transport raises `MCPError` on a missing subprocess pipe.** The four
  `assert proc.stdin/stdout is not None` checks across `_read_loop`, `_answer_server_request`, `request`, and
  `notify` are now guards that raise `MCPError` when the subprocess was started without the expected pipe.
- **Genuine never-happens invariants are documented and marked, not removed.** The remaining `assert`s — a
  value guaranteed non-`None` by an adjacent caller guard, a model validator, or the immediately-preceding
  assignment (e.g. `MemoryEngine._vectors_for`'s embedder, the governance ledgers' `_load` store, the optimize
  evaluators' bound dataset, `Choreography._bind_step`'s binder, `SemanticLayer._measure_sql`'s aggregation) —
  are kept as a cheap type-narrowing aid, each with a clarifying comment and a justifying `# noqa: S101`.

### Added

- **`vincio._assert_robustness` — the static `-O`-robustness gate.** It scans every public module for an
  `assert` statement and holds the whole tree to **zero** unmarked ones: `# noqa: S101` (the standard "use of
  `assert`" code) is the per-site marker affirming a reviewed never-happens invariant, the way `# noqa: BLE001`
  marks a reviewed broad `except`. There is no frozen baseline — the inline marker is the accepted form, so a
  new load-bearing `assert` fails the build the moment it lands unless it is converted to a guard. An `assert`
  inside a docstring example is a string literal, not a statement, and is correctly ignored. Reproduce offline
  with `python -m vincio._assert_robustness`.
- **VincioBench `hygiene` family gains the 6.5 checks.** `assert_robustness_conformant` (clean tree **and** the
  detector provably bites on an injected bare `assert` while ignoring a marked one), `assert_robustness_clean`,
  and `assert_robustness_gate_detects_tamper`, plus the `hygiene_unmarked_asserts` (gated to `0`) and
  `hygiene_marked_asserts` stats. Three new SLOs (`hygiene_assert_robustness_*`) are published and held by
  at-least-as-strict budgets.

## [6.4.0] - 2026-06-30

**Docstring / behaviour parity — the hardening line (6.x) continues.** A standing internal audit found
docstrings that advertised behaviour the code no longer performed, a handful of stale spec-numbering comments,
and one example whose guarantee no longer fired. A docstring that promises behaviour the code does not perform
is a quiet lie that misleads a reader who trusts it; this release makes each one either true or corrected and
re-derives the reconciliation from the live code so it cannot drift back. Additive: the frozen top-level
contract `vincio.__all__` is unchanged, so `API_VERSION` stays `5.0`, the `vincio migrate 6.x` codemod table
stays empty, and a clean upgrade needs zero source changes.

### Changed

- **`vincio.context.budgeting` no longer advertises a `redistribute` reclaim that nothing invoked.** The module
  docstring described reclaiming an under-used block's tokens for evidence and memory, but the allocator already
  performs the meaningful redistribution at allocation time — fixed-size blocks are charged at cost and the
  whole remainder is distributed across the flexible blocks (evidence, memory, tool results) proportionally to
  their fractions. The docstring now says exactly that, and the verified-dead `BudgetAllocator.redistribute`
  static method (referenced nowhere) is removed.
- **`vincio.context.llmlingua`'s faithfulness docstring matches the gate.** It claimed
  `vincio.optimize.compression_tuning` gated adoption on `compression_faithfulness` / `faithfulness_preserved`,
  but the tuner reads a `faithfulness` eval metric. The docstring now separates the two truths accurately:
  `compression_faithfulness` / `faithfulness_preserved` are the offline measures of answer-bearing survival
  (the same fidelity check VincioBench uses), and `CompressionTuner` (`app.gate_compression`) gates installing
  the compressor on answer quality, a `faithfulness` metric floor, and a verifiable token saving over the eval
  suite.
- **`MemoryEngine.delete` delegates to `forget`.** The two were near-duplicate bodies; `delete` is now the
  unannotated form of `forget` (one body, with a plain delete recording no reason and `forget` recording one),
  so the deletion path has a single implementation. `forget`'s `reason` widens to accept `None` (the form
  `delete` calls in); the audit semantics of both are unchanged.
- **Stale comments cleared.** The context-compiler pipeline step numbering is monotonic again (`# 7` budget
  allocation, `# 8` selection — was `# 8` / `# 7+8`), the `vincio/input/*` module and function docstrings drop
  the external-spec "item N" numbering, and the redundant second clause of the `prompts/lint.py` PROMPT001
  vague-role check is removed (behaviour unchanged).

### Fixed

- **The federated default-deny consent demonstration now fires every run.** In `examples/13_data_and_analytics.py`
  the consent ledger persisted its grant to the app's on-disk store, so a second run loaded the grant and the
  default-deny refusal silently stopped firing — and the example smoke test, which runs each example once in a
  fresh temp directory, never caught it. The demonstration now attaches a store-less `ConsentLedger`, so it
  starts empty every run: the first contribution is refused for want of an `ANALYTICS` grant, then granted and
  admitted, deterministically regardless of any consent persisted on disk.

### Added

- **HygieneBench docstring-parity SLOs.** The `hygiene` VincioBench family folds in `docstring_parity_conformant`
  (with `docstring_parity_budgeting` / `docstring_parity_compression` / `docstring_parity_consent` /
  `docstring_parity_memory`), each re-deriving a reconciled claim from the live code — the dead reclaim method
  stays gone and the allocator distributes the whole remainder; the tuner reads the `faithfulness` metric its
  docstring names; a store-less default-deny ledger refuses then admits; and `delete` delegates to `forget` with
  the audit details preserved — held by `budgets.json` and published in `slos.json` / `docs/reference/slo.md`.
  Guarded by `tests/test_docstring_parity.py`, including a regression that runs the federated section twice
  against a persisted store and asserts the refusal both times.

## [6.3.0] - 2026-06-30

**Wire-or-retire — the hardening line (6.x) continues.** A standing internal audit found capabilities that
were implemented and even public, but that nothing could reach — no `app.*` verb, no example, no internal
caller — so dead weight read as supported API. This release wires each one to a production path (or, where the
primitive is a deliberate advanced deep-import API, documents it as such) and adds a guard that fails the build
if a wired capability ever loses its entry point. Additive: the frozen top-level contract `vincio.__all__` is
unchanged (the new entry points are `ContextApp` verbs and one subpackage-public result type), so `API_VERSION`
stays `5.0`, the `vincio migrate 6.x` codemod table stays empty, and a clean upgrade needs zero source changes.

### Added

- **`app.retrieve_facts`** — reasoning retrieval as a first-class verb. Instead of one top-k by query
  similarity, a `FactSchema` declares the facts a task needs and the engine retrieves the task query, then runs
  a targeted retrieval for each fact still uncovered, returning a typed `FactRetrieval` (merged evidence,
  per-fact `FactCoverage`, and a `complete` flag that stays false while a *required* fact is missing — the
  insufficient-evidence signal an agent acts on). `FactRetrieval` is exported from `vincio.retrieval`.
- **`app.consolidate_memory`** — the episodic→semantic tier transition as a maintenance verb. Pass a
  `session_id` to consolidate one session now, or omit it to sweep every session whose episodes have all aged
  past `min_age_days` (the periodic form, wired through the new `MemoryEngine.promote_aged_episodes`). Schedule
  it from your own job runner; Vincio runs no background loop of its own.
- **`ModelProvider.exact_token_counter`** — the hook the token-counter registry is wired through. A provider
  that can count a model's tokens exactly and offline registers its counter when it is built (the OpenAI
  provider via `tiktoken`; an in-process GGUF model via its own tokenizer), so counting is model-id-driven
  through `register_token_counter` rather than tied to one global default. `register_provider_token_counters`
  performs the (idempotent) registration at provider init.
- **Wire-or-retire guard** (`vincio._wire_or_retire`). A static guard holding a frozen ledger of the
  formerly-unhooked capabilities: every listed capability must resolve to a live reach (an `app.*` verb, an
  engine method, a registration helper, or a public class member) and — for a wired one — be referenced by
  production code outside its defining module, so a capability cannot silently become dead surface again.
  Guarded by `tests/test_wire_or_retire.py`; reproduce offline with `python -m vincio._wire_or_retire`.
- **HygieneBench wire-or-retire SLOs.** The `hygiene` VincioBench family folds in `wire_or_retire_conformant`
  (with `wire_or_retire_clean` and a `wire_or_retire_gate_detects_tamper` "the gate bites" proof, plus a
  `hygiene_wired_capabilities` count), held by `budgets.json` and published in `slos.json` /
  `docs/reference/slo.md`.

### Changed

- **`use_context_governor` backs cold-span paging with a blob store.** `use_context_governor(blob_store=…)`
  (or `evidence_store=…`) wires `BlobEvidenceStore` into the long-horizon governor, so a compacted span's full
  text pages back from a content-addressed blob store — across a restart or another worker — instead of only a
  process-local store. The default (in-memory) path is unchanged.
- **`vincio.core.tokens` docstring corrected.** It no longer claims Vincio ships Anthropic / Gemini exact
  counters; it accurately describes `register_token_counter` as the extension point a provider plugs an exact,
  offline counter into (a hosted provider whose only exact count is a network round-trip ships none, since that
  round-trip is unsuitable for the per-candidate scoring loop).
- **`compile_streaming` / `recompile` / `CompileStreamEvent` documented as advanced API.** These remain
  deep-import primitives on `ContextCompiler` (a full run already streams end-to-end through `app.stream`); the
  performance guide and the advanced-context example now frame and demonstrate them as such.

## [6.2.0] - 2026-06-30

**Observable failure — the hardening line (6.x) continues.** A best-effort fallback that catches a broad
`Exception` and continues is correct policy — a broken embedder or a rejected memory write must never break a
run — but several did it *silently*, with no log and no metric, so a real bug could hide inside the fallback. A
standing internal audit found them. This release makes every such fallback observable and adds a lint that
forbids a new silent one. Additive and surface-preserving: the frozen top-level contract `vincio.__all__` is
unchanged, so `API_VERSION` stays `5.0`, the `vincio migrate 6.x` codemod table stays empty, and a clean
upgrade needs zero source changes.

### Added

- **Observable-failure primitive** (`vincio.core.diagnostics`). `note_suppressed(label, *, level=, detail=)`
  records a suppressed best-effort failure in one call: it logs the suppression on a dedicated diagnostics
  channel (`vincio.suppressed`, capturing the active exception's traceback) and increments a process-wide
  counter keyed by a stable label. An operator both *watches* the failures (enable the channel at `DEBUG`) and
  *scrapes their rate* (`suppressed_failure_counts()`) without the fallback ever breaking the run it guards. The
  counter and channel are an observability surface, not a stable API contract.
- **Observable-failure checker** (`vincio._observable_failure`). A static, AST-based scanner of every public
  module for a *broad* `except` (`Exception` / `BaseException` / a bare `except:` / a tuple containing one) and
  for `contextlib.suppress(Exception)`. It enforces one always-on invariant, the way `vincio._error_contract`
  enforces the error contract: the whole public tree carries **zero** unmarked silent swallows — a broad
  handler must re-raise, record its failure observably (a logger call or `note_suppressed`), or carry a
  justifying `# noqa: BLE001` (the inline marker the codebase already uses for a reviewed, deliberately-silent
  swallow). There is no frozen baseline; the inline marker is the per-site accepted form. Guarded by
  `tests/test_observable_failure.py`; reproduce offline with `python -m vincio._observable_failure`.
- **HygieneBench observable-failure SLOs.** The `hygiene` VincioBench family folds in
  `observable_failure_conformant` (with `observable_failure_clean` and an
  `observable_failure_gate_detects_tamper` "the gate bites" proof, plus a `hygiene_silent_swallows` count),
  held by `budgets.json` and published in `slos.json` / `docs/reference/slo.md`.

### Changed

- **Silent best-effort swallows made observable.** The fallbacks that previously swallowed a broad exception
  with no log or metric now call `note_suppressed` with a stable label, across `core/runtime.py` (lineage /
  fertility record, content marking, region resolution), `core/app.py` (provable-erasure store deletes and
  cache invalidation, the SBOM attach in `certify`), `core/tokens.py` (counter build fallbacks),
  `context/compiler.py` (the embedding→lexical fallback), `context/longhorizon.py` (the guarded-memory write),
  `data` (`engagement` / `federated` artifact serialization and re-bind, `analysis` follow-up planning),
  `governance/cards.py`, `cli/doctor.py`, `plugins.py`, `providers/registry.py`, the `a2a` / `mcp` request
  dispatchers (at `WARNING`), and `storage` / `settlement` best-effort persistence and probes. Behavior is
  unchanged — each fallback still continues — but the suppression is now logged and counted. A handful of
  fallbacks that surface the error into a returned result (a `401` response, a denied permission decision, a
  failed-step outcome, a `PeerVisit` record) carry a justifying `# noqa: BLE001` instead.

## [6.1.0] - 2026-06-30

**Error-contract conformance — the hardening line (6.x) continues.** Vincio's contract is that every error it
raises derives from `VincioError`, so an application catches the whole family with one `except VincioError`
and branches on the stable `.code`. A standing internal audit found three public entry points that leaked a
bare built-in instead. This release converts them and makes the contract *mechanical*, on the same
freeze-and-gate idiom 6.0 used for the two-level public surface. Additive and surface-preserving: the frozen
top-level contract `vincio.__all__` is unchanged, so `API_VERSION` stays `5.0`, the `vincio migrate 6.x`
codemod table stays empty, and a clean upgrade needs zero source changes.

### Added

- **Error-contract checker** (`vincio._error_contract` + `docs/reference/error-contract.txt`). A static,
  AST-based scanner of every public module for a bare built-in raise on a public entry point (every enclosing
  function and class public; dunders like `__init__` count). It enforces two invariants, the way
  `vincio._surface` enforces the two-level surface: the `ContextApp` (`app.*` verb) surface is held to **zero**
  off-contract raises by an always-on check that needs no allowlist, and the full classified baseline of
  accepted public built-in raises (internal input-validation, abstract-base placeholders, the `AttributeError`
  a `__getattr__` must raise) is frozen in a committed manifest, so a new public built-in raise is a deliberate,
  reviewed edit. Guarded by `tests/test_error_contract.py`; regenerate with
  `python -m vincio._error_contract --freeze`.
- **HygieneBench error-contract SLOs.** The `hygiene` VincioBench family folds in `error_contract_conformant`
  (with `error_contract_app_verbs_clean`, `error_contract_frozen`, and an `error_contract_gate_detects_tamper`
  "the gate bites" proof), held by `budgets.json` and published in `slos.json` / `docs/reference/slo.md`.

### Changed

- **Three off-contract built-in raises converted to typed `VincioError` subclasses.** All three carry their
  catalog `.code` and remediation; `except VincioError` (the documented contract) was always the correct catch
  and is unaffected:
  - `ContextApp.atest_time_search` / `test_time_search` now raise **`InputError`** (was `ValueError`) for an
    unknown `strategy`.
  - `register_error_locale` (`vincio.core.error_catalog`) now raises **`ConfigError`** (was `KeyError`) for a
    code not in the catalog. Code that caught the bare `KeyError` specifically should catch `ConfigError` (or
    `VincioError`) instead.
  - `MultimodalEmbedder._multimodal_payload` — reachable from the public `embed()` / `embed_multimodal()` on a
    non-`abc.ABC` base — now raises **`ConfigError`** naming a concrete subclass (was `NotImplementedError`).

## [6.0.0] - 2026-06-30

**Public-surface hygiene — the opening of the hardening line (6.x).** The 5.1–5.4 fit-and-finish line made
the complete platform honest, fast, navigable, and one-line easy on its *surface*; the hardening line turns
that discipline *inward*, onto the codebase's interior. Every change here is additive, dependency-free, and
**surface-preserving**: the frozen top-level contract `vincio.__all__` is unchanged (540 symbols), so
`API_VERSION` stays `5.0`, the `vincio migrate 6.x` codemod table is empty, and a clean upgrade needs zero
source changes. None of it changes a result.

### Added

- **Two-level surface-consistency gate** (`vincio._surface` + `docs/reference/subpackage-surface.txt`).
  `vincio.__all__` is the frozen top level, but each public subpackage also declares its own `__all__` — and
  that surface had drifted and was undeclared. The gate enumerates every public subpackage's `__all__`,
  classifies each symbol `TOP` (re-exported in `vincio.__all__` — the same object), `DUP` (an intentional
  top-level name collision, e.g. the tabular `vincio.data.Dataset` beside the eval `vincio.Dataset`), or `SUB`
  (subpackage-only public), and freezes the classified result in a committed manifest. Three invariants are
  enforced (`tests/test_surface_consistency.py`): every name resolves to a live attribute (no dead surface),
  no duplicate/malformed entries, and the surface matches the manifest byte-for-byte — so any `__all__` change
  is a deliberate, reviewed edit, exactly as `public-surface.txt` freezes the top level. Regenerate with
  `python -m vincio._surface --freeze`.
- **HygieneBench (`families.hygiene`)** — a new offline, deterministic VincioBench family folding the gate's
  guarantees into a headline `surface_consistency` SLO (with `surface_dead_symbol_free`, `surface_frozen`, and
  a `surface_gate_detects_tamper` "the gate bites" proof), held by `budgets.json` and published in `slos.json`
  / `docs/reference/slo.md`, the way `docs_conformance` keeps the docs graph honest.
- **Two public exceptions declared.** `IdentityError` and `GovernanceVerificationError` — both raised in
  shipped, reachable code and present in the error catalog — are added to `vincio.core.errors.__all__` (they
  were defined and raised but undeclared). The top-level surface still re-exports only the `VincioError` base,
  unchanged.

### Removed

- **Nine verified-dead public symbols**, each referenced nowhere in `vincio/`, `tests/`, `examples/`, or
  `benchmarks/` (dead surface that read as supported API): `vincio.core.concurrency.race_with_timeout`,
  `vincio.core.tokens.CallableTokenCounter`, `vincio.core.shapley.ashapley_values`,
  `vincio.core.utils.truncate_text`, `PromptSpec.build_ast` (superseded by the stable/volatile AST split),
  the `vincio.input.classifiers.LLMTaskClassifier` type alias (never wired into `classify_task` / `route`),
  `TwoStageIndex.stats`, the unused `SpeculativePrefetcher(reranker=...)` constructor parameter, and
  `MemoryEngine.for_tenant` (tenant-scoped memory remains reachable via `remember(tenant_id=...)`). None was
  part of the frozen top-level `vincio.__all__`; the surface-consistency gate now prevents dead surface from
  silently returning.

### Changed

- **The hardening line's opening, on the developer surface** (already landed, now released). The fluent `Flow`
  was refactored onto a single shared *lowering* module (`vincio.tasks._lowering`) so a one-line task and its
  `Flow` twin emit identical builder calls by construction — steps are typed, self-applying values, and a flow
  clones through one frozen config. The 26 examples were consolidated to **16, one per macro-feature** — the
  entire data & analytics plane is now a single offline tour (`examples/13_data_and_analytics.py`) — and the
  capability map, `llms.txt`, and every Related block were regenerated with the docs-graph gate green.

## [5.9.0] - 2026-06-30

Notebook-native analysis surface — **the fourth and final phase of the data & analytics extension line
(5.6–5.9), completing the plane and the platform.** The governed, cited, offline-verifiable analysis you run
in a script now runs **interactively**, without becoming a hosted notebook service. The existing
`enable_rich_reprs` extends to the data artifacts — a `QueryResult`, an `AnalysisResult`, a `Chart`, and a
sealed `DataNarrative` render inline as cards with clickable cell citations, the lineage verdict, and the
audit id — and a `notebook_session(app, ...)` threads register → query → analyze → chart → cite through the
*same* governed primitives a script calls (it is a thin front over `app.data_engagement`), sealing into the
same signed, audited `DataNarrative`. Entirely additive — two new top-level symbols and a richer
`vincio.notebook` module; `API_VERSION` stays `5.0`, the `vincio migrate 5.x` codemod table stays empty, and
a clean upgrade needs zero source changes. Dependency-free and offline, like the rest of the plane.

### Added

- **Cited inline reprs for the data plane.** `enable_rich_reprs` now also attaches pure, offline
  `_repr_html_` / `_repr_markdown_` to a `QueryResult`, an `AnalysisResult`, a `Chart`, and a `DataNarrative`.
  A query result renders as a table whose cells are tooltipped with the source cells they rest on and whose
  clickable disclosure lists every cell citation; an analysis renders each cited finding; a chart shows its
  content-bound credential and source cells; a narrative shows its stage chain, the structural integrity
  verdict recomputed from the bytes, and the audit id it was sealed under. The reprs are pure — they only ever
  surface an artifact's real, verifiable facts and never re-execute or fabricate.
- **`notebook_session(app, ...)` → `NotebookSession`** — a thin, interactive front over
  `app.data_engagement`. Each verb (`register`, `profile`, `sample`, `screen`, `query`, `analyze`, `chart`,
  `query_metric`, `cite`) delegates to the *same* governed primitive a script calls, renders the artifact
  inline, and threads it into the engagement's hash-linked narrative. `session.narrative` seals the same
  signed, audited `DataNarrative` a script produces; `session.verify()` recomputes the whole chain *and*
  re-executes every inline finding against the content-hashed source (`data_bound`), and a tampered source
  flips the verdict — so a notebook exploration is governed, reproducible, and offline-verifiable by
  construction.
- **Two new public symbols** exported at the top level and from `vincio.notebook`: `notebook_session` and
  `NotebookSession`. Public surface 538 → 540. The data-artifact repr helpers (`query_result_html`,
  `analysis_result_html`, `chart_html`, `data_narrative_html`, and their `_markdown` twins) are added to
  `vincio.notebook.__all__`.
- **DataPlaneBench** gains a `notebook` family — **repr-faithfulness** (every repr surfaces the artifact's
  content hash and exact citations, and a tampered stage flips the narrative's integrity verdict) and
  **notebook-session-verifies** (a threaded session seals a data-bound, signed narrative, refuted against a
  tamper) — held by two new SLOs in `docs/reference/slo.md`.
- **Docs & examples:** a fully-offline `examples/notebooks/06_notebook_native_analysis.ipynb`, the
  notebook-native section in `SECURITY.md`, the published SLOs, and synchronized `README` / `ROADMAP` /
  `llms.txt` / generated API index and the public-surface freeze.

### Changed

- **`vincio/notebook.py`** — `enable_rich_reprs` / `disable_rich_reprs` now bind the four data artifacts in
  addition to the core result types; the binding list is shared so enabling and disabling stay symmetric.
- **`benchmarks/vinciobench.py`** — `bench_data_plane` gains a `notebook` sub-key; `ROADMAP.md` (5.9 moved
  from *Planned* to *What ships today*, the extension line declared shipped in full), `README.md`,
  `SECURITY.md`, `benchmarks/README.md`, `llms.txt`, and the generated API index / public-surface freeze are
  synchronized.

## [5.8.0] - 2026-06-30

Forecasting & causal-inference verifier kernels — **the third phase of the data & analytics extension line
(5.6–5.9).** The verified-reasoning plane already gives an *arithmetic*-class claim a checkable certificate;
this gives the **statistical** claims a data answer makes — a trend, a correlation, a confidence / prediction
interval, a forecast — the same, on the same `Certificate` surface. Each kernel **recomputes** the stated
statistic from the **cited cells** and refutes one the data does not bear out, and the correlation kernel
**refuses a correlation stated as causation** that earns no warrant — no controls, or a confounder that, once
partialled out, collapses the association. Entirely additive — a new `vincio.verify.statistical` module and a
`statistical_claims=` argument on the existing `app.verify_reasoning`; `API_VERSION` stays `5.0`, the
`vincio migrate 5.x` codemod table stays empty, and a clean upgrade needs zero source changes. The
deterministic kernels are dependency-free; an optional CAS backend sits behind `vincio[verify]`.

### Added

- **`vincio.verify.statistical` — four deterministic statistical kernels** on the `ReasoningVerifier`
  protocol: **`TrendVerifier`** recomputes an ordinary-least-squares slope / intercept and `R²` goodness-of-fit
  and checks a stated trend and direction; **`CorrelationVerifier`** recomputes a Pearson correlation and, for
  a causal claim, demands a warrant; **`IntervalVerifier`** recomputes a Student-t confidence interval for a
  mean or an OLS prediction interval at a point; **`ForecastVerifier`** re-runs a declared deterministic model
  (`naive` / `mean` / `drift` / `linear` / `moving_average` / `ses`) and checks the projection.
- **Spurious-causation refutation.** A `CorrelationClaim` with `causal=True` is refused when it declares no
  controls and no randomized design, and refuted when its declared confounders, once partialled out via the
  **partial correlation** (recomputed from the cited series), collapse the association below a threshold — so a
  confounded correlation reported as a cause is a deterministic refutation, while a genuine controlled
  association and a randomized-design claim are verified.
- **Statistics bound to cited cells.** A `CitedSeries` carries the `CellRef`s its values came from (build one
  from a cell-cited `QueryResult` with `CitedSeries.from_cells(result.citations(row, col))`); a value swapped
  after it was cited makes the series unbound and the kernel refuses, so a smuggled number cannot ride a real
  citation.
- **`app.verify_reasoning(..., statistical_claims=[...])`** adds the statistical kernels to the default set
  automatically and records the verdict on the audit chain as before; a refuted statistical claim refuses to
  emit and a `regenerate` callback may repair it by returning a corrected `StatisticalClaim`, which re-grounds
  the context before re-certifying — the refuse-or-repair loop over *statistical reasoning*.
- **Deterministic statistics core** (pure, offline, dependency-free): `ols_fit`, `pearson_r`,
  `partial_correlation`, `mean_confidence_interval`, `prediction_interval`, `forecast`, and Student-t
  quantiles (`student_t_ppf`) via the regularized incomplete beta function — no NumPy / SciPy.
- **Optional `CasTrendVerifier`** (`vincio.verify.smt`, behind `vincio[verify]`) re-discharges the OLS trend
  fit with exact `sympy.Rational` arithmetic — zero floating-point drift in the recomputation.
- **Twelve new public symbols** exported at the top level and from `vincio.verify`: `CellRef`, `CitedSeries`,
  `StatisticalClaim`, `TrendClaim`, `CorrelationClaim`, `IntervalClaim`, `ForecastClaim`, `TrendVerifier`,
  `CorrelationVerifier`, `IntervalVerifier`, `ForecastVerifier`, and `statistical_verifiers`. Public surface
  526 → 538. `VerificationContext` gains a `statistical_claims` field.
- **VerifiedReasoningBench** gains `statistical_soundness` and `refutes_spurious_causation` (folded into the
  existing `certificate_soundness`), held by a new **refutes-spurious-causation** SLO alongside the
  certificate-soundness SLO in the `verified_reasoning` family.
- **Docs & examples:** `examples/25_statistical_certificates.py` (a fully-offline walk of all four kernels
  built from a cell-cited query result), a statistical-claims section in the verified-reasoning guide, the
  published SLOs in `docs/reference/slo.md`, and synchronized `README` / `ROADMAP` / `SECURITY` / `AGENTS` /
  `llms.txt` / capability map and the public-surface freeze.

### Changed

- **`benchmarks/vinciobench.py`** — `bench_verified_reasoning` certifies the statistical kernels and the
  spurious-causation refutation; `README.md`, `ROADMAP.md` (5.8 moved from *Planned* to *What ships today*),
  `SECURITY.md`, `AGENTS.md`, `llms.txt`, and the generated API index / public-surface freeze are synchronized.
- **`app.verify_reasoning`** gains an optional `statistical_claims` keyword; the default behaviour with no
  statistical claims is byte-identical (the statistical kernels are added only when claims are supplied).

## [5.7.0] - 2026-06-30

Cross-org / federated analytics — **the second phase of the data & analytics extension line (5.6–5.9).**
A governed metric run across **more than one organization's** data without pooling the raw rows into a
shared warehouse — the analytics analogue of federated self-improvement and the data-plane twin of
`CrossOrgEngagement`. The query is negotiated as a `Contract`, choreographed as a `Saga` whose steps run
each org's governed query plane **locally** and return only the aggregated, cell-cited `MetricResult`, and
reconciled into one signed, offline-verifiable `FederatedNarrative` whose every finding re-derives from each
org's content-hashed source. Entirely additive — a new `vincio.data.federated` module and one app method
behind it; `API_VERSION` stays `5.0`, the `vincio migrate 5.x` codemod table stays empty, and a clean
upgrade needs zero source changes. Dependency-free, deterministic, and offline.

### Added

- **`FederatedQuery` — the shape of a cross-org governed metric** (`vincio.data.federated`): the measures
  and dimensions, the source columns it touches, the residency posture, the budget, and a `min_members`
  k-anonymity contributor floor. Its `digest()` is bound into a negotiated `Contract`'s hashed `scope`, so
  the agreed query shape is tamper-evident. `validate_against` refuses a non-partition-decomposable measure
  (`AVG` / `COUNT_DISTINCT` / ratio) at construction with guidance to federate its decomposable components.
- **`FederatedDataEngagement`** (`app.federated_data_engagement`) — a governed, compositional facade over
  several `FederatedMember`s (each an org id bound to its *own* `ContextApp`). `negotiate` binds the query
  into a signed `Contract`; `dispatch` choreographs a contract-governed `Saga`, one step per member, each
  running that org's `query_metric` **locally** and returning only the aggregated `MetricResult` — the raw
  rows are never serialized into the dispatch, the journal, or the narrative; `reconcile` combines the
  aggregates into one `FederatedFinding` per metric and group. `run` threads all three.
- **Exact reconciliation** for the partition-decomposable aggregations: `SUM` and `COUNT` add across orgs,
  `MIN` / `MAX` take the extremum, group by group — so a federated total is the true total. Every org must
  compute the metric by the **same** layer definitions (an org whose layer digest differs is refused), and
  the grouping attributes are the layer's governed *dimensions*, never raw identifiers.
- **`FederatedNarrative` — signed, hash-chained, offline-verifiable**: the ordered chain of
  `FederatedStage`s (negotiate → choreograph → per-org query → reconcile) carrying the reconciled findings.
  `verify()` recomputes the chain from the bytes alone (a re-ordered stage, an edited digest, a tampered
  head, or a forged signature is caught), and `FederatedDataEngagement.verify` additionally re-executes each
  member's aggregate against its content-hashed source and re-derives every reconciled value — so a tamper
  to any org's source or to the reconciliation is caught even when the chain is intact. Sealed onto the
  audit chain (action `federated_data_engagement`).
- **Governance crosses the boundary intact:** residency-aware egress refusal (reusing `ResidencyPolicy`),
  the consent ledger's `ANALYTICS` purpose, and the differential-privacy accountant apply to a member's
  contribution exactly as they would to a local query — a member outside the posture, without consent, or
  over its budget is **refused and audited** (action `federated_query_governance`) — and a round below the
  `min_members` contributor floor is refused so a single org is never singled out.
- **`FederatedQuery`, `FederatedMember`, `FederatedContribution`, `FederatedFinding`, `FederatedStage`,
  `FederatedSignature`, `FederatedVerification`, `FederatedNarrative`, and `FederatedDataEngagement`** are
  exported at the top level and from `vincio.data`. Public surface 517 → 526.
- **DataPlaneBench / `federated_analytics`** family with three SLOs — **rows-never-cross** (a per-row
  sentinel reaches neither the saga journal nor the sealed narrative), **federated-data-binding** (every
  finding re-derives from each org's source and the reconciled totals equal the brute-force totals over the
  pooled rows, with a tampered reconciliation caught), and **governance-preservation** (residency, consent,
  the privacy budget, and the contributor floor each refuse a non-compliant round).
- **Docs & examples:** `examples/24_federated_analytics.py` (a fully-offline two-org walk),
  `docs/concepts/federated-data-engagement.md`, a federated-analytics section in the analyze-data guide, the
  capability-map topic, and the published SLOs in `docs/reference/slo.md`.

### Changed

- **`benchmarks/vinciobench.py`** — `bench_data_plane` gains the `federated_analytics` sub-family;
  `README.md`, `ROADMAP.md` (5.7 moved from *Planned* to *What ships today*), `SECURITY.md`, `AGENTS.md`,
  `llms.txt`, and the generated API index / public-surface freeze are synchronized.

## [5.6.0] - 2026-06-30

Real-time & streaming analytics — **the first phase of the data & analytics extension line (5.6–5.9).**
The profiling, query, governed-metric, and quality plane re-expressed over an **unbounded event stream**
rather than a bounded `Dataset`, computed one **window** at a time so the working set stays invariant to
the event volume, with every per-window answer citing the exact events it rests on and re-deriving offline
against the captured window. Entirely additive — a new `vincio.data.streaming_analytics` module and one
new app method behind it; the frozen surface grows by three top-level symbols (`StreamWindow`,
`EventCitation`, `WindowedQueryResult`), `API_VERSION` stays `5.0`, the `vincio migrate 5.x` codemod table
stays empty, and a clean upgrade needs zero source changes. Dependency-free, deterministic, and offline.

### Added

- **`StreamWindow` — the windowing policy** (`vincio.data.streaming_analytics`): `tumbling` / `sliding` /
  `session` windows over an unbounded `RowStream`, windowed by **event time** (a numeric `time_column`) or
  processing time, partitioned by `key_by`, and closed on a **watermark** past the window end plus an
  allowed `lateness`. A late event for an already-closed window is **dropped and counted**, never silently
  misfiled; a stream more out of order than the lateness (or with unbounded key cardinality) is **refused**
  by `max_open_windows` rather than growing the working set without bound.
- **Windowed analogues of the batch primitives** — each a lazy iterator emitting one result per closed
  window: `StreamWindow.profile` → `WindowedProfile`, `query` → `WindowedQueryResult` (over the
  read-only-verified query plane), `query_metric` → `WindowedMetricResult` (over the governed semantic
  layer), `screen` → `WindowedQualityReport` (`DataQualityRails` per window, with per-violation offending
  event offsets), and a bounded-memory `aggregate` → `WindowedAggregation` (riding `stream_aggregate`).
- **Event-level provenance & offline verification:** each closed window is captured into a bounded
  `CapturedWindow` (the events as a schema-bearing `Dataset`, their stable stream offsets, the bounds, a
  content hash). A windowed answer cites the exact source **events** via `EventCitation`
  (`stream@<offset>!<column>`, the streaming analogue of `CellCitation`), and `verify()` re-executes against
  the captured window, confirming the answer and every cited event re-derive from the bytes — a tampered
  captured event is caught.
- **`app.stream_analytics(window)`** → a governed `StreamingAnalytics` driver: audits every emitted window
  on the hash-chained audit log (action `stream_window`), screens any natural-language question on the same
  injection rail, and drives a **live** async source (a queue, a websocket, a realtime session's events) via
  `drive(...)` exactly as it replays a log — with an `on_window` callback and `max_windows` bound for an
  alerting rule.
- **`StreamWindow`, `EventCitation`, and `WindowedQueryResult`** are exported at the top level (the full set
  — `WindowKind`, `CapturedWindow`, the five `Windowed*` results, and `StreamingAnalytics` — from
  `vincio.data`). Public surface 514 → 517.
- **DataPlaneBench / `realtime`** family with three SLOs — **windowed-correctness** (the windowed group-by
  equals the brute-force ground truth), **bounded-memory** (footprint invariant as the stream grows 100×),
  and **incremental-provenance** (every window verifies offline, every cited offset is in-window, a tampered
  event is caught) — plus an events/s throughput budget.
- **Docs & examples:** `examples/23_realtime_streaming_analytics.py` (a fully-offline walk over a replayed
  event log and a live async feed), `docs/concepts/realtime-streaming-analytics.md`, a streaming-analytics
  section in the analyze-data guide, the capability-map topic, and the published SLOs in
  `docs/reference/slo.md`.

### Changed

- **`benchmarks/vinciobench.py`** — `bench_data_plane` gains the `realtime` sub-family; `README.md`,
  `ROADMAP.md` (5.6 moved from *Planned* to *What ships today*), `SECURITY.md`, `AGENTS.md`, `llms.txt`,
  and the generated API index / public-surface freeze are synchronized.

## [5.5.0] - 2026-06-30

The developer-experience capstone — **rebuild the on-ramp to match the platform behind it.** The
library is complete and the docs are a connected graph (5.4), but the first surfaces a developer or an
agent meets — the README, the examples, and the agent-facing context files — had drifted and stayed
feature-tour-shaped. 5.5 makes the on-ramp first-class, and completes the ROADMAP by scheduling the
formerly-exploratory data & analytics extensions as concrete phases (5.6–5.9). Entirely additive:
docs and examples only, **no public symbol changes** (`vincio.__all__` and `API_VERSION` untouched),
dependency-free, deterministic, and offline.

### Added

- **A three-tier examples suite:**
  - **`examples/notebooks/`** — five Google Colab-ready notebooks (quickstart, RAG, agents & tools,
    evaluation, data analysis): one `pip install`, offline by default, an *Open in Colab* badge each.
    Gated by `tests/test_example_notebooks.py` (valid JSON, an install cell, no stale outputs, and the
    code cells run offline end to end).
  - **`examples/applications/`** — real-world small backends: a FastAPI **grounded-RAG service**, a
    **ticket-triage API** (typed output + scoped memory + an approval-gated escalation tool), a
    **structured-extraction service** (bounded self-correction), and a no-framework **CLI research
    agent**. Each FastAPI app splits an offline-testable `core.py` from a thin FastAPI `main.py`.
    Gated by `tests/test_example_apps.py` (cores run offline; the FastAPI shells are exercised with a
    test client when FastAPI is installed).
- **`docs/guides/analyze-data.md` was already added in 5.4**; 5.5 adds the notebooks/applications
  index pages and threads them through `examples/README.md`.

### Changed

- **`README.md` fully redesigned** — current version/test badges, a Colab "try it in 30 seconds"
  callout, an honest offline story (pass `MockProvider()` or set a provider+key; the default provider
  is OpenAI), the three example tiers, and the scheduled forward plan.
- **`AGENTS.md` rewritten** — restructured contributor map with the examples & docs systems, a
  CI-skip note for optional-dependency code, and an "adding a subsystem" checklist.
- **`llms.txt` generator (`vincio/_docmap.py`) rewritten** — a richer, accurate preamble and gotchas
  (the OpenAI-default-provider / explicit-mock reality, the data-plane app methods, the three example
  tiers); still regenerated from `vincio.__all__` and gated for freshness.
- **`ROADMAP.md` completed** — the open-ended "Exploring — later" section is replaced by a scheduled
  data & analytics extension line (**5.6** real-time & streaming analytics, **5.7** cross-org /
  federated analytics, **5.8** forecasting & causal-inference verifier kernels, **5.9**
  notebook-native analysis surface), and the developer-experience overhaul lands in *What ships
  today*.

### Fixed

- Corrected the "offline by default" claim in the README and `llms.txt`: a bare `ContextApp` uses the
  OpenAI default and needs a key; the offline path is `provider=MockProvider()` (which auto-generates
  schema-valid output). The examples and notebooks use this explicitly.

## [5.4.0] - 2026-06-30

The fourth and final fit-and-finish minor on the frozen 5.x platform — **make the docs navigable.** Every subsystem ships
a concept, a guide, a reference entry, and a runnable example, but those ~80 leaf pages were held together by one
hand-ordered index and little else: the 16 concept pages had zero inbound links, no data-plane concept reached the example
that demonstrates it, the curated `api.md` documented only a fraction of the public `app.*` methods, and the completeness
gate only checked that a literal substring appeared *somewhere* in the docs. 5.4 adds the **connective tissue, not a new
domain**: a single source of truth (`vincio/_docmap.py`, rendered the way `vincio/_apiref.py` renders `api-generated.md`)
binds every public `app.*` verb to the concept that explains it, the guide that applies it, the example that demonstrates
it, and the reference anchor that specifies it, grouped by the six capability facades — and from it generates the capability
map, a single-sourced Related cross-link block on every concept and guide, a staged learning path, the `api.md` app-method
index, and `llms.txt`. Entirely additive and backward-compatible: **no public symbol changes** (`vincio.__all__` and
`API_VERSION` are untouched), dependency-free, deterministic, and offline.

### Added

- **`vincio/_docmap.py`** — the doc graph: one reviewable source of truth (a `Topic` taxonomy) binding every public
  `ContextApp` method to its concept / guide / example / reference, grouped by the six capability facades (`runs` /
  `knowledge` / `governance` / `optimization` / `serving` / `training`). It renders the capability map, the Related blocks,
  the learning path, the `api.md` app-method index, and `llms.txt`, and exposes the docs-graph checks. Dependency-free and
  on the `mypy --strict` ladder.
- **`docs/reference/capability-map.md`** (generated) — every public `app.*` verb bound to the page that documents it.
- **`docs/learning-path.md`** (generated) — a staged getting-started → grow-into-depth spine replacing the flat "Next
  steps" list; `getting-started.md`'s next steps now lead with it.
- **A single-sourced `Related` cross-link block** on every concept and guide page, so a reader traverses laterally instead
  of returning to the index.
- **`vincio docs` CLI verb** — `vincio docs map [--check]` regenerates the artifacts (or gates freshness), `vincio docs
  check` runs the docs-graph check, and `vincio docs serve` previews the docs locally (HTML rendering via the new
  `vincio[docs]` extra, raw Markdown otherwise). The map and the check run dependency-free.
- **The `vincio[docs]` extra** — bundles the richer Markdown renderer (`markdown-it-py`) behind `vincio docs serve`; the
  capability map and the coverage check never require it.
- **The `docs_conformance` VincioBench family** with three published SLOs — link integrity, capability-map coverage, and
  navigation reachability — plus companion budgets (every concept connected, no orphans, `llms.txt` current, the gate
  bites), proven by a fully-offline `examples/22_connected_docs.py`.
- **Concise docstrings** for 15 previously-undocumented core `ContextApp` methods (`run`, `arun`, `configure`, `evaluate`,
  `add_evaluator`, `add_validator`, `add_optimizer`, `add_memory`, `set_policy`, `workflow`, `stats`, `resolve_provider`,
  `principal_for`, `aclose`, `acited_report`), so the capability map and `llms.txt` carry a summary for every verb.
- **`docs/guides/analyze-data.md`** — a task-oriented guide for the data & analytics plane, the home the data-plane concept
  pages now reach.

### Changed

- **`docs/reference/api.md`** now carries a generated, completeness-gated app-method index (every public `app.*` method,
  grouped by facet), and the docs index links the capability map and the learning path.
- **`llms.txt`** is now regenerated from `vincio.__all__` and the doc graph and gated for freshness, the way
  `api-generated.md` and the error catalog are — no longer hand-maintained with no test references.
- **The docs-completeness gate** deepens from a substring check into a docs-graph check (links resolve, every concept
  reaches a guide + example + reference anchor, every `app.*` method appears in `api.md`, no orphans, `llms.txt` current)
  in `tests/test_docs_graph.py`, bridged from `tests/test_docs_completeness.py`.

## [5.3.0] - 2026-06-29

The third fit-and-finish minor on the frozen 5.x platform — **make the power one-line easy.** The platform is
feature-complete, but its power was broad: a `ContextApp` carries a couple hundred methods, so the five jobs a newcomer
actually has — grounded RAG Q&A, a tool-using agent, structured extraction, an eval, and a multi-step flow — each took a
fistful of string-keyed builder calls (the canonical RAG path is six coupled calls) where an LCEL chain, a LlamaIndex query
engine, a DSPy module, or a Haystack pipeline each cost one line. 5.3 adds the missing **top layer, not a new capability**:
a small, discoverable `vincio.tasks` namespace of task-shaped constructors plus one fluent, immutable `Flow`, each a
**purely-compositional facade** that configures a `ContextApp` with sane governed defaults using the *same* public builder
calls a caller would make by hand — so the one-liner **lowers to the exact same governed `ContextApp.run` packet** as the
verbose form, proven byte-identical. Entirely additive and backward-compatible (the new symbols are tagged `@experimental`
and `API_VERSION` stays `"5.0"`), dependency-free, deterministic, and offline.

### Added

- **The `vincio.tasks` namespace** — six one-line, task-shaped entry points, also re-exported at the top level
  (`from vincio import rag, extractor, tool_agent, evaluation, chat, Flow`) and tagged `@experimental`:
  - `rag(sources, ...)` → a `RagTask` that indexes sources, turns on grounding-only answering with citations, and adds the
    groundedness + citation-accuracy evaluators; `.ask(question)` runs a full grounded, cited, eval-scored run.
  - `extractor(schema, ...)` → an `Extractor`; `.extract(text)` returns a validated Pydantic object.
  - `tool_agent(tools=, writes=, approve=, ...)` → a `ToolAgent`; write tools are denied by default and surfaced as pending
    approvals, `.run(task)` drives the governed model+tool loop.
  - `evaluation(dataset, metrics=, gates=, ...)` → an `Evaluation`; `.run()` evaluates the bound dataset.
  - `chat(...)` → a re-presentation of `app.assistant` (a session-aware, multi-turn `Assistant`).
  - `Flow` → a fluent, immutable pipeline (retrieve → ground → call → validate → evaluate); every step returns a new Flow,
    and `.run(input)` lowers the whole pipeline to one governed run. `.app` on every facade is the escape hatch to all of
    `ContextApp`'s deep methods (none shadowed, none unreachable).
- **A shared lowering harness** (`vincio.testing.lowering`: `selection_signature`, `result_signature`, `run_signature`) that
  projects a compile or a finished run to a deterministic signature (the packet `spec_hash` plus the stable outputs). The
  single-pass feature arena's selection-byte-identity check (5.2) and the new ergonomic byte-identical proof now share this
  one harness; the duplicated `_selection_signature` copies in `tests/` and `benchmarks/` were factored out to it.
- **The ErgonomicsBench VincioBench family** with three published SLOs — conciseness (each use-case is one entry point,
  benchmarked head-to-head in `benchmarks/competitive.py` against LCEL, the LlamaIndex query engine, DSPy, and Haystack),
  compiles-byte-identical (the ad-hoc form lowers to the verbose form's packet and result), and escape-hatch-total — each
  held by an at-least-as-strict CI budget.
- **`examples/00_one_liners.py`** (sits before the quickstart) and **`docs/concepts/ergonomic-surface.md`**, mapping each
  one-liner to the deep methods it composes and proving the byte-identical lowering offline.

### Changed

- `vincio.__all__` gains six additive `@experimental` symbols (`rag`, `extractor`, `tool_agent`, `evaluation`, `chat`,
  `Flow`); the frozen public surface (`docs/reference/public-surface.txt`) and the generated API index are re-frozen to
  match. The 5.x SemVer contract (`API_VERSION = "5.0"`) is unchanged and every existing call-site is untouched.

## [5.2.0] - 2026-06-29

The second fit-and-finish minor on the frozen 5.x platform — **make the default compile path faster, honestly.**
The win is not a new capability but the *same* selection made measurably cheaper on the default, dependency-free path
VincioBench actually gates (NumPy is an optional accelerator, never present for a gated number). The compile hot path
(`_normalize` → pre-filter → `_remove_duplicates` → `_resolve_conflicts` → `_select`) re-derived each candidate's stemmed
terms, shingles, and similarity-blocking tokens pass after pass; the bounded global `lru_cache` on those derivations
thrashes on the 10k+ pools the streaming pre-filter exercises, so the O(n²) dedup / conflict / selection constant factor
was paid many times over. 5.2 derives each candidate's features **exactly once per compile** and threads them through every
pass, returning **byte-identical context** — the same selection, made faster. Entirely additive and backward-compatible (no
public symbol changes; `API_VERSION` stays `"5.0"`), dependency-free, deterministic, and offline.

### Added

- **A per-compile feature arena** (`vincio/context/features.py`, `FeatureArena`) that derives each candidate's stemmed
  terms, word shingles, and similarity-blocking tokens **once per compile** — in an unbounded per-compile memo discarded
  when the compile finishes — and threads them through scoring, dedup, conflict, and selection, so a 10k-scale pool that
  overruns the bounded global cache pays each derivation once. The features are byte-identical to the per-pass derivation,
  and the arena is a fresh per-compile object, so the shared compiler stays concurrency-safe. Ships behind a
  `ContextCompilerOptions.single_pass_selection` flag (and a `performance.single_pass_selection` config knob), on by
  default, exactly the way `reuse_candidate_set` does.
- **A norm-cached cosine** (`vincio.retrieval.embeddings.cosine_with_norms` / `vector_norm`). The semantic dedup / conflict
  / selection passes compute each embedding's L2 norm once and reuse it across every pairwise comparison instead of
  recomputing both norms per call. Bit-for-bit identical to `cosine`, which now delegates to it.
- **A batched token counter** (`vincio.core.tokens.count_tokens_many`) used by the compiler's normalization pass; resolves
  the counter once for a batch and is element-for-element identical to `count_tokens` per item, with a fast path for a
  native batch tokenizer.
- **A bounded BM25 top-k.** `BM25Index.search` selects its top-k with `heapq.nlargest` (O(n log k)) instead of a full sort
  (O(n log n)), returning the identical hits a full sort would — the bounded result equals the prefix of the full ranking.
- **PerfBench gates** holding the optimization: `selection_byte_identical`, `vectorized_selection.equivalent`, and
  `retrieval.topk_identical` (`eq:true`) prove it never changes *what* is selected, and a `single_pass.compile_speedup`
  **ratio floor** (published 1.05×, gated 1.08×) makes an erased win fail the build rather than passing silently under a
  loose latency ceiling. Four published SLOs, each held by an at-least-as-strict CI budget; the committed
  `benchmarks/profile_stages.py --compare` reports the before/after per-stage breakdown.

### Changed

- The context compiler builds a per-compile scorer carrying the feature arena (and any semantic embedding vectors) rather
  than threading per-compile state onto the shared scorer, so concurrent compiles never alias. The shared, state-free
  scorer is still used unchanged when the single-pass flag is off.
- `vincio.context.compiler._block_tokens` is derived once per kept candidate in the dedup pass instead of twice; this
  applies whether the feature arena is on or off and is selection-preserving.

## [5.1.0] - 2026-06-29

The first fit-and-finish minor on the frozen 5.x platform — **make the cost report honest.** The data-driven
`ModelRegistry` was already the single source of truth the cost `PriceTable`, the capability guard, the cost/latency
router, the model cascades, and the energy/carbon accounting all read from; it was simply under-populated and un-gated, so
a current-lineup model (the OpenAI o-series and `gpt-4.1` families, OpenAI/Mistral embeddings, and the entire
`openai_compat` preset surface) resolved to nothing, warned once, and then billed $0, and a stale price shipped green
because the registry version was a hand-bumped string with no enforced horizon. 5.1 keeps the exact `ModelProfile` shape
and the exact exact→alias→longest-prefix lookup path and only **fills and holds them**. Entirely additive and
backward-compatible (no public symbol changes; `API_VERSION` stays `"5.0"`), dependency-free, deterministic, and offline.

### Added

- **A shipped, reviewable model catalog.** The inline `_builtin_catalog()` is lifted into
  `vincio/providers/model_catalog.json` behind the existing loader, populated with the real current lineup of every
  provider Vincio supports: OpenAI o-series (`o1` / `o3` / `o4-mini`) + `gpt-5` + `gpt-4.1` + `text-embedding-3-*`,
  Anthropic `claude-3-5`/`3-7-sonnet` + `3-5-haiku` + `3-opus` beside the 4.x / Fable tier, Mistral
  medium / codestral / pixtral / `mistral-embed`, the `openai_compat` presets priced instead of $0
  (groq / together / fireworks / openrouter / deepseek / perplexity / xai / nvidia headline models), and Google reconciled
  to verified live reality — `gemini-3-*` marked preview-not-GA, `gemini-2.5-flash` the stable GA default,
  `gemini-embedding-001` the live embedder.
- **`ModelProfile.priced_as_of`** — an additive, optional ISO date stamping when each profile's pricing was last verified.
- **An `as_of`-deterministic freshness horizon.** `CATALOG_RELEASED` and `FRESHNESS_HORIZON_DAYS` evaluate price freshness
  against the catalog's *release* date, never the wall clock, so a frozen release reports the same verdict forever and only
  a genuinely stale snapshot fails the gate.
- **`ModelRegistry.coverage_report()`** — a deterministic, offline drift detector returning a `RegistryCoverageReport`
  proving every supported provider's default and capability-heuristic families and every `openai_compat` preset resolve to
  a non-sparse, priced profile, that no GA billable model of a paid provider silently bills $0, that no price has drifted
  past the freshness horizon, and — an explicit no-routing-drift check — that the canonical router / cascade / energy picks
  are unchanged by the refresh.
- **`vincio registry coverage`** and **`vincio registry sync`** CLI verbs. `coverage` runs the drift detector and exits
  non-zero on a gap; `sync` (behind the new `vincio[registry-sync]` extra) is **review-only** — it diffs a provider's live
  `list_models()` into a candidate overlay for a human to price and merge, and never mutates the shipped catalog.
- **A `registry_coverage` VincioBench family** with three published SLOs — coverage, freshness, and no-silent-$0 — each
  held by an at-least-as-strict CI budget, plus a runnable [`examples/21_model_pricing_registry.py`](examples/21_model_pricing_registry.py).

### Changed

- `OpenAICompatPreset` gains a `default_model` naming each gateway's priced headline model. Behavior is otherwise
  unchanged — any other model id on a gateway is still yours to choose and warns once via `ModelUnknownWarning` if
  uncatalogued, rather than silently billing $0.

## [5.0.0] - 2026-06-29

The data & analytics capstone, and the second long-term-support major. Seven rungs (4.1–4.7) delivered the data plane's
primitives — first-class tabular evidence and the compact encoder, profiling / sampling / fit-in-window and the quality
rails, governed text-to-query with cell-level provenance, the multi-step analysis agent, content- and data-bound charts,
streaming out-of-core processing, and the semantic layer's governed metrics — each grounded, cited, and offline-verifiable
on its own. 5.0 unifies them: a `DataEngagement` facade (`app.data_engagement`) threads the whole plane behind one governed,
audited call-path into a content-bound, signed, hash-chained `DataNarrative` that verifies offline from the bytes alone and
is **data-bound** (every captured finding re-executes against the content-hashed source and re-derives), the analytics
analogue of `CrossOrgEngagement`. It is then the next consolidation major: `API_VERSION` is promoted to `"5.0"`, the public
surface — expanded additively across the 4.x line — is re-frozen (503 → 508 public symbols), the empty `vincio migrate 5.0`
codemod table ships, and the data & analytics plane is declared **feature-complete and frozen**. Entirely additive and
backward-compatible (a clean 4.x → 5.0 upgrade needs zero source changes), dependency-free, deterministic, and offline.

### Added

- **`DataEngagement` — the data & analytics plane threaded as one system.** `app.data_engagement(*, dataset="",
  question="", analyst=None)` returns a purely-compositional facade whose lifecycle methods — `register`, `profile`,
  `sample`, `fit`, `screen`, `query`, `analyze`, `chart`, `query_metric`, and `cite` — each delegate to the *same* `app.*`
  primitive a caller would use directly (each unchanged and still usable on its own), capture the artifact they produced
  (exposed as `eng.result` / `eng.analysis` / `eng.chart_` / `eng.metric` / …), and record it as a stage in one
  hash-linked narrative. `cite` assembles the findings into a per-figure data-bound cited deliverable through the existing
  cited-report builder. `eng.record_stage(stage, artifact, *, binder=None, **summary)` is an escape hatch for any other
  artifact.
- **`DataNarrative` — a content-bound, signed, hash-chained narrative of a whole engagement.** `eng.seal(*, sign=True,
  record_audit=True)` mints it and lands it on the hash-chained audit log (action `data_engagement`); each `DataStage`
  binds the lifecycle verb, the artifact's own published commitment (`result_hash` / `chart_hash` / `layer_hash`), and a
  digest of its bytes into a link chaining to the previous one. `DataNarrative.verify(verifier=None, *, require=None,
  artifacts=None)` recomputes the whole chain from the bytes alone — a re-ordered stage, an edited digest, a broken link, a
  tampered head, or a forged signature is caught (`broken_at` pinpoints the first failing stage). `to_wire` / `from_wire` /
  `require_valid` / `print_summary`.
- **Data-binding — every analytical finding re-derives from the source it cites.** `eng.verify(verifier=None, *,
  require=None, catalog=None)` re-digests every captured artifact against its bound digest *and*, given the live catalog
  (defaulting to `app.data_catalog()`), re-executes every captured query, analysis, chart, and metric against the
  content-hashed source and confirms each re-derives from the bytes (surfaced as `DataEngagementVerification.data_bound`),
  so a tampered source is caught even when the chain itself is intact — the analytics analogue of a generated report's
  per-claim entailment, applied to the whole engagement.
- **New public symbols (5):** `DataEngagement`, `DataNarrative`, `DataStage`, `DataEngagementSignature`, and
  `DataEngagementVerification`, re-exported from `vincio` and `vincio.data`.
- **`data_analysis_conformance` VincioBench family** with end-to-end, data-bound, and tamper-evident SLOs, and a
  whole-pipeline competitive comparison (`benchmarks/competitive.py::bench_data_engagement`) against pandas-ai, the
  LlamaIndex query engine, the LangChain SQL agent, Vanna, and native DuckDB. A runnable example
  (`examples/20_data_engagement.py`).

### Changed

- **`API_VERSION` promoted to `"5.0"`** and the package version to `5.0.0`. The public surface is **re-frozen** for the 5.x
  line: `docs/reference/public-surface.txt` pins the exact 508-name surface and a build gate fails on any silent drift.
- **`vincio migrate 5.0`** ships with an empty rename table (the 4.x line was additive end to end, so a clean 4.x → 5.0
  upgrade needs zero source changes); see [`MIGRATION.md`](MIGRATION.md).
- The data & analytics plane is **feature-complete and frozen** under the [stability policy](docs/reference/stability.md):
  no further data-plane *primitive* is scheduled; subsequent data-plane work is bug-fix and standards-tracking only.

### Removed

- Nothing. The deprecation runway was empty across the entire 4.x line, so no public API reached its `removed_in` runway.

## [4.7.0] - 2026-06-29

The semantic layer & governed metrics — the seventh rung of the data & analytics plane: define the analytical vocabulary
**once** (measures, dimensions, and derived columns) so a natural-language question maps to a *governed metric* rather than
a raw column and is computed **one way everywhere**. A `SemanticLayer` compiles a metric to a single canonical read-only
`SELECT` and runs it through the *existing* governed query plane, so the answer is cell-level cited and offline-verifiable;
`MetricResult.verify` additionally proves the SQL was the layer's canonical compilation, so an ad-hoc number cannot pass as
the governed one. Column-level data lineage carries a metric's provenance — and a subject's right-to-erasure — into the
dataset plane. Additive in `vincio.data` (new `vincio/data/semantic.py`) and `vincio.governance`; entirely
backward-compatible, dependency-free, deterministic, and offline. `API_VERSION` is unchanged at `"4.0"` — 4.7 extends the
4.x surface additively (501 → 503 public symbols).

### Added

- **`SemanticLayer` — measures, dimensions, and derived columns defined once.** Built over one registered table from a
  `DerivedColumn` (a row-level calculation, `revenue = price * qty`, that composes), a `Dimension` (a groupable attribute, a
  column or an expression), and a `Measure` — the **governed metric**: an `Aggregation` (`sum` / `avg` / `min` / `max` /
  `count` / `count_distinct`) over a column or derived column, an optional row filter, or a **ratio** of two other measures
  (`avg_order_value = total_revenue / orders`). Names share one namespace; a duplicate or non-identifier name, a
  derived-column or ratio-measure cycle, a measure declaring neither an aggregation nor a complete ratio, or an expression
  that could break out of its clause is refused with a `SemanticLayerError`, and every metric is dry-run-grounded against the
  table at definition. Build it declaratively or with the chaining `add_derived` / `add_dimension` / `add_measure` helpers.
- **`query_metric` / `MetricQuery` / `MetricResult` — a question becomes a governed metric.** `SemanticLayer.query` (and the
  free `query_metric` / `app.query_metric`) resolves a metric name, a list of names, a `MetricQuery`, or a natural-language
  question (grounded by metric/dimension names and synonyms, injection-screened first, returning nothing rather than guessing)
  to the governed measure, compiles it to **one canonical read-only `SELECT`** (derived columns inlined, ratios compiled
  zero-safe with `NULLIF`), and runs it through the existing read-only-verified query plane — so the metric is computed one
  way everywhere (two phrasings compile to byte-identical SQL), cell-level cited, and offline-verifiable. `MetricResult`
  wraps the cited `QueryResult` with the `MetricQuery` and the layer's content hash; `MetricResult.verify` proves the
  definitions are unchanged, the SQL **is** the layer's canonical compilation (an ad-hoc query is rejected), and the result
  re-derives from the hashed source.
- **`MetricLineage` — column-level provenance.** `SemanticLayer.column_lineage` (and `app.metric_lineage`) resolves a metric
  through its derived-column graph and any ratio references to its **base columns**, the underlying measures, the governed
  aggregate SQL, and the **source** the dataset was ingested under.
- **Lineage & right-to-erasure reach the dataset plane.** `app.register_dataset(..., source=)` records the dataset in the
  `LineageIndex` under its source with its columns; `LineageRecord` gains `datasets` / `dataset_columns` and the index gains
  `record_dataset` / `source_of_table` / `datasets_for`. `app.erase_source` now drops a source's registered datasets (and any
  semantic layer over them) from the data catalog, counted in `ErasureResult.datasets_removed` and recorded in the signed
  `ErasureProof` (`removed_ids["datasets"]`). `DataCatalog.remove` drops a table.
- **App surface.** `app.semantic_layer(table, *, measures=, dimensions=, derived=, name=, register=, validate=)` defines and
  (by default) registers a layer; `app.query_metric(request, *, layer= / table=, by=, where=, order_by=, descending=, limit=,
  dataset=, ...)` computes a governed metric (audited `metric_query`); `app.metric_lineage(metric, *, layer= / table=)` reports
  its column-level provenance (audited `metric_lineage`). Defining a layer is audited (`semantic_layer_define`).
- **Top-level surface.** `SemanticLayer` and `query_metric` are re-exported at the package top level; the rest (`Measure`,
  `Dimension`, `DerivedColumn`, `Aggregation`, `MetricQuery`, `MetricResult`, `MetricLineage`) lives in `vincio.data`. New
  error `SemanticLayerError` (its own `SEMANTIC_LAYER_ERROR` code, inheriting `DataError`).
- **DataPlaneBench / semantic-layer.** A `semantic_layer` section of the **DataPlaneBench** family adds a governed-one-way SLO
  (a metric compiles to one canonical SQL and returns the same number however the question is phrased), a metric-verifiable
  SLO (the result re-derives from the bytes while a forged ad-hoc result and a tampered source both fail), and a
  lineage-reaches-the-dataset-plane SLO (a metric's lineage resolves to its base columns and source, and an erasure sweep
  removes the dataset and records it in the proof). The metric compiles to the same SQL a hand-written query would, so there
  is no new performance-sensitive path.

### Documentation

- New concept page **[The semantic layer and governed metrics](concepts/semantic-layer-and-governed-metrics.md)** (under
  `docs/`), a runnable **`examples/19_semantic_layer_governed_metrics.py`**, and updates to the README, `llms.txt`,
  `SECURITY.md`, `AGENTS.md`, the SLO reference, and the ROADMAP (the 4.7 row moves from planned to shipped).

## [4.6.0] - 2026-06-29

Streaming & out-of-core bulk processing — the sixth rung of the data & analytics plane: process a dataset *far larger
than memory* in **bounded passes**, inside a footprint that does not grow with the row count. A lazy, re-iterable,
schema-bearing `RowStream` is profiled, fitted, sampled, aggregated, and encoded — each a single bounded pass;
`stream_aggregate` is a bounded-memory group-by that tracks the number of *groups*, not rows; `encode_stream` renders the
compact encoding header-once (optionally gzipped); the context compiler gains a **streaming candidate pre-filter** that
bounds a 10k+ evidence pool before full scoring; and `stream_map` runs an analytical transform over a stream at scale
through the existing `BatchRunner`. Additive in `vincio.data` (new `vincio/data/streaming.py`), `vincio.context`, and
`vincio.core`; entirely backward-compatible, dependency-free, deterministic, and offline. `API_VERSION` is unchanged at
`"4.0"` — 4.6 extends the 4.x surface additively (499 → 501 public symbols).

### Added

- **`RowStream` — a lazy, re-iterable, schema-bearing out-of-core handle.** The out-of-core analogue of a `Dataset`: it
  never materializes its rows, holding a factory that produces a fresh row iterator on demand. Build one from records, rows,
  a dataset, a re-iterable generator factory, or a file read line by line — `RowStream.from_csv` / `.from_jsonl` /
  `.open(path, format=)`. The CSV reader infers each column's type from a bounded peek and coerces losslessly (a value that
  does not round-trip exactly stays text); a bare generator object is single-use, while a sequence, a zero-argument callable,
  or a file supports the multiple passes the operators need. `.chunks(size)` yields bounded `Dataset` chunks; `.profile()` /
  `.fit(max_tokens=)` / `.sample(k)` reuse the existing `profile_stream` / `fit_stream` / `reservoir_sample` single-pass
  kernels; `.aggregate(...)`, `.encode(compress=)`, and `.materialize()` round it out.
- **`stream_aggregate` / `StreamAggregation` — a bounded-memory group-by.** Group a stream by one or more columns and reduce
  measures (`sum` / `mean` / `min` / `max`; each group's row `count` always emitted) over each group in a single pass. The
  working set holds one accumulator per distinct *group*, never the rows, so a table far larger than memory aggregates inside
  a fixed footprint; a group cardinality beyond `max_groups` raises `StreamError` rather than growing without bound. The
  result is a small `Dataset` (`to_dataset()` / `to_evidence_item()`).
- **`encode_stream` — streaming compact encoding and compression.** Render a stream to the compact, lossless encoding
  header-once, row-by-row, so a dataset larger than memory is encoded in one bounded pass. `compress=True` gzip-compresses the
  output; a binary `sink=` streams the bytes straight to a file. The streaming header omits the row count, and
  `decode_table` reads rows to end-of-input, so the round-trip stays exact. New low-level kernel helpers
  `tabular.encode_header` / `tabular.encode_row`.
- **Streaming candidate pre-filter in the context compiler.** `ContextCompilerOptions.max_candidates` (and the config field
  `performance.max_context_candidates`): when the evidence pool exceeds the cap, a single streaming pass keeps the top-N by a
  cheap lexical relevance proxy (a bounded heap) and drops exact duplicates by a bounded content fingerprint *before* the full
  multi-signal scoring, the O(n²) dedup and conflict passes, and any embedding materialization — so the expensive stages and
  the resident vector footprint never see more than the cap as a 10k+ corpus grows. Every drop is recorded in the excluded
  report (`prefiltered_low_relevance` / `prefiltered_duplicate`); a `compiler.prefilter_drops` counter surfaces the count.
  Off by default (`None`), so an unbounded compile is byte-for-byte unchanged.
- **`stream_map` / `BulkMapResult` — analytical pipelines at scale on the `BatchRunner`.** Run a per-chunk transform over a
  stream by chunking it into the existing provider Batch API (half-cost, bounded concurrency): each bounded chunk becomes one
  request via `build_request(chunk, index)`, the set is dispatched, and the responses are reconciled by chunk index. A missing
  or failed chunk surfaces as a failed result rather than being dropped.
- **App surface.** `app.stream_dataset(source, *, schema= / columns=, name=, format=)` opens a stream from a file, records,
  rows, or a dataset; `app.aggregate_stream(...)` runs the group-by; `await app.map_stream(...)` runs the BatchRunner map
  (defaulting to the app's own provider). Thin wrappers over the `vincio.data` functions.
- **Top-level surface.** `RowStream` and `stream_aggregate` are re-exported at the package top level; the rest
  (`StreamAggregation`, `encode_stream`, `stream_map`, `BulkMapResult`) lives in `vincio.data`. New error `StreamError`
  (inherits the `DATA_ERROR` code, so `except DataError` catches it).
- **DataPlaneBench / streaming.** A `streaming` section of the **DataPlaneBench** family adds a throughput SLO (rows/s through
  the streaming group-by, plus a tokens/s floor through the streaming encoder), a memory-stays-bounded SLO (the aggregation's
  peak resident set for a 100×-larger stream stays within a small factor of the smaller one), and a pre-filter-bounds-pool SLO
  (a 10,000-candidate corpus compiled under a cap keeps only the cap's worth while every relevant item survives).
  `benchmarks/competitive.py` gains a `streaming` comparison (the streaming group-by's peak memory vs materialize-then-aggregate).

### Documentation

- New concept page **[Streaming and out-of-core bulk processing](concepts/streaming-and-out-of-core.md)** (under `docs/`), a
  runnable **`examples/18_streaming_out_of_core.py`**, and updates to the README, `llms.txt`, `SECURITY.md`, `AGENTS.md`, the
  SLO reference, and the ROADMAP (the 4.6 row moves from planned to shipped).

## [4.5.0] - 2026-06-29

Charts & cited analytical artifacts — the fifth rung of the data & analytics plane: a cited query result becomes an
analytical *artifact* a reader can trust. `generate_chart` turns a `QueryResult` into a spec-driven `Chart` that is
**content-bound** (a C2PA *data-driven* credential bound to its rendered bytes, exactly the provenance a generated image
carries) and **data-bound** (a back-reference to the exact source cells, verified by re-executing the source query and
re-binding the credential). The cited-report builder extends to **figures**, so an analytical deliverable is per-claim
entailed *and* per-figure data-bound. Additive in `vincio.data` and `vincio.generation`; entirely backward-compatible,
dependency-free, deterministic, and offline. `API_VERSION` is unchanged at `"4.0"` — 4.5 extends the 4.x surface
additively (494 → 499 public symbols).

### Added

- **`generate_chart` / `app.generate_chart` — spec-driven charts.** Turn a cited `QueryResult` (or an `AnalysisResult` /
  bare `Dataset`) into a `Chart`. The encoding is inferred from the result schema when not pinned — a dimension on the x
  axis, a measure on the y axis, a second dimension as the color series — a temporal x axis defaults to a line, everything
  else to a bar, and the mark vocabulary is `bar` / `line` / `point` / `area` / `arc`. `app.generate_chart` runs a
  natural-language question or SQL string through the governed query plane first when given one, and lands the run on the
  hash-chained audit log (`chart_generate`).
- **`Chart` — content-bound and data-bound, offline-verifiable.** Carries the `ChartSpec`, the rendered `data` bytes and
  their media type, the C2PA `ProvenanceManifest` bound to those bytes, and the lineage back to the source rows
  (`provenance`, `source_hashes`, `result_hash`, `chart_hash`). `cite_refs()` returns the exact source-cell locators the
  figure rests on (`sales#r0!revenue`); `content_bound()` checks the credential still binds the bytes; `verify(catalog)`
  re-executes the source query, confirms the plotted figure is a faithful projection of that verified result, and confirms
  the credential binds the bytes — an edited spec, an edited byte stream, a stripped credential, or a tampered source flips
  it to `False`. `to_evidence_item()` projects the figure into cited table evidence; `save(path)` writes the bytes (with a
  `.c2pa.json` sidecar for the JSON spec).
- **`VegaLiteRenderer` (default, dependency-free) and `MatplotlibRenderer` (behind the new `vincio[charts]` extra).** The
  default renderer emits a portable Vega-Lite v5 JSON spec with the data embedded inline — no drawing library; the
  matplotlib renderer rasterizes the same spec to a PNG that carries its credential embedded the way a generated image
  does. `ChartRenderer` is the pluggable protocol.
- **`mark_data_driven_content` — a content credential for data-driven media.** The C2PA marker sibling of
  `mark_synthetic_content` for charts, rendered tables, and data exports: it binds the bytes by SHA-256 the same way but
  carries the IPTC `dataDrivenMedia` digital-source-type and `is_synthetic=False` — a faithful rendering of real values is
  not synthetic content. Exported from `vincio.governance`.
- **The cited-report builder extends to figures.** `Figure` (`Figure.from_chart` / `Figure.from_table`) embeds a chart or a
  table into a report: it gets a `[F1]`-style marker the narrative can reference, is rendered into the document, and — when
  a catalog is supplied — is verified to re-derive from its source. `CitedReportBuilder.build_report(...)` and
  `app.cited_report(...)` / `acited_report(...)` take `figures=` and an optional `catalog=` (the app default resolves the
  registered datasets). `CitationContract(require_figure_binding=True, min_figure_binding_rate=)` gates per-figure data
  binding — a non-deriving figure raises `CitationValidationError`, the per-figure analogue of per-claim entailment.
  `CitationCoverage` gains `figures` / `data_bound_figures` / `figure_binding_rate`; `CitedReport` gains `figures`
  (`FigureBinding` verdicts).
- **Top-level surface.** `generate_chart`, `Chart`, `ChartSpec`, `ChartType`, and `Figure` are re-exported at the package
  top level; the full surface (`ChartEncoding`, `ChartChannel`, `ChartRenderer`, `VegaLiteRenderer`, `MatplotlibRenderer`,
  `FigureBinding`) lives in `vincio.data` / `vincio.generation`. New error code `CHART_ERROR` (`ChartError`).
- **DataPlaneBench / charts.** A `charts` section of the **DataPlaneBench** family adds a `data_bound` SLO (the figure
  re-derives and a tampered source is caught), a `figure_cited` SLO (the figure cites the exact source cells, aggregates
  included), and a `content_bound` SLO (the credential binds the bytes and an edited byte stream is caught), all gated
  `true`.

### Documentation

- New concept page **[Charts and cited analytical artifacts](concepts/charts-and-cited-artifacts.md)** (under `docs/`), a
  runnable **`examples/17_charts_cited_artifacts.py`**, and updates to the README, `llms.txt`, `SECURITY.md`, the SLO
  reference, and the ROADMAP (the 4.5 row moves from planned to shipped).

## [4.4.0] - 2026-06-29

The data-analysis agent & multi-step EDA — the fourth rung of the data & analytics plane: a bounded analysis agent that
plans, queries, inspects, and refines over a dataset through the *existing* governed query plane, producing a **cited
analytical narrative** that re-derives from the bytes the way a cited report does. The exploration is bounded by an
explicit budget, every step runs read-only-verified, and the verifier is the query plane's offline re-execution, not a
model. Additive in the `vincio.data` subpackage; entirely backward-compatible, dependency-free, deterministic, and
offline. `API_VERSION` is unchanged at `"4.0"` — 4.4 extends the 4.x surface additively (491 → 494 public symbols).

### Added

- **`analyze_dataset` / `app.analyze_data` — the data-analysis agent.** Runs a bounded loop — plan → query → inspect →
  refine → synthesize — over a registered dataset: a deterministic, schema-grounded plan (an overview, the objective
  grounded by the same `HeuristicQueryPlanner` text-to-query uses, each measure's extreme and total, a
  measure-by-dimension breakdown, and a drill into the group that dominates) where **every step runs through the governed,
  read-only-verified query plane**, so a finding is grounded by construction, never hallucinated. The objective is screened
  by the same injection detector the text rails use (a refusal raises `UnsafeQueryError`), and the run lands on the
  hash-chained audit log (`data_analysis`, with a `deny` entry pinpointing a refused objective).
- **`AnalysisResult` — a cited analytical narrative, offline-verifiable.** Carries the `narrative` (one cited finding per
  line), the executed `AnalysisStep`s (each with its `query`, `finding`, exact-cell `cite_refs`, and `coverage`), a
  content-bound `result_hash`, and `answer()` / `primary_step()` accessors. `verify(catalog)` re-executes every step and
  confirms the narrative and every cited cell re-derive from the bytes — a tampered source or a tampered narrative flips it
  to `False`. `to_evidence_item()` projects the analysis into cited table evidence; `render("report")` shows each finding's
  underlying query. The narrative's coverage is the weakest across its steps and is always stated, never silently
  downgraded.
- **`AnalysisBudget` — bounded by construction.** Caps the total step count (`max_steps`), the refinement drill-downs
  (`max_refinements`), each query's result (`max_rows`), and the breakdown fan-out — there is no open-ended search. The
  agent reuses the existing query plane for grounding and verification rather than growing a parallel search stack.
- **`AnalysisAgent` — the app-wired agent.** Resolves the catalog (the app's registered datasets or a one-shot
  `dataset=`), screens the objective, audits the run, and — when a model is configured — may ask it for additional
  analytical follow-up questions, each still grounded and verified by the query plane (offline, or when the model returns
  nothing groundable, the agent is byte-for-byte the deterministic core).
- **`DuckDbQueryEngine` — execution at scale.** A drop-in `QueryEngine` (behind the new `vincio[data]` extra) that runs the
  *same verified read-only SQL* on DuckDB, re-asserting read-only at the engine boundary. It reports result-level lineage
  (the result still re-derives on `verify()`); the offline `sqlite3` engine remains the path that derives per-cell
  citations.
- **Top-level surface.** `AnalysisResult`, `AnalysisAgent`, and `analyze_dataset` are re-exported at the package top level;
  the full surface (`AnalysisBudget`, `AnalysisStep`, `AnalysisStepKind`, `DuckDbQueryEngine`, `AnalysisError`) lives in
  `vincio.data`. New error code `ANALYSIS_ERROR`.
- **DS-1000 / InfiAgent-DABench / DABench adapters** (`vincio.evals.DS1000Adapter`, `InfiAgentDABenchAdapter`,
  `DABenchAdapter`, with `ds_1000_tasks_from_export` / `infiagent_dabench_tasks_from_export` / `dabench_tasks_from_export`
  and shipped fixtures) scored by **task success at budget** — the agent's answer matches the gold answer and the analysis
  finishes within its step budget. An `analysis` section of the **DataPlaneBench** family adds a `success_at_budget` SLO, a
  `narrative_cited` SLO, and a `verifiable` SLO, all gated `true`.

### Fixed

- **Cell-level lineage for a query with a string-literal `WHERE`.** The query plane's lineage witness (the rowid rebuild
  for a projection / filter, and the group witness for a group-by aggregation) reused clause text that had its string
  literals blanked for keyword scanning, so a `WHERE col = 'value'` produced empty cell citations. Clause text is now
  sliced from the original SQL (literals intact) while keyword boundaries are still scanned on a length-aligned mask, so a
  filtered or grouped query with a string literal now carries correct cell-exact provenance. (Queries with numeric `WHERE`
  predicates or no `WHERE` were unaffected; this surfaced through the analysis agent's drill-downs.)

### Documentation

- New concept page **[Data-analysis agent and multi-step EDA](concepts/data-analysis-agent.md)** (under `docs/`), a runnable
  **`examples/16_data_analysis_agent.py`**, and updates to the README, `llms.txt`, `SECURITY.md`, the SLO reference, and the
  ROADMAP (the 4.4 row moves from planned to shipped). The 4.3 text-to-query concept's "what it is not" note is updated,
  since the multi-step data-analysis agent now ships.

## [4.3.0] - 2026-06-28

Governed text-to-query & cell-level provenance — the third rung of the data & analytics plane: the core analyst
capability. A question over a registered dataset becomes a query that is *verified before it runs* — schema-grounded,
read-only by default, and cost-bounded — executed by the standard-library `sqlite3` engine where the data lives rather
than by pouring rows into the prompt, and answered with a citation to the **exact cells** it rests on, offline-verifiable
the way a cited report is. Additive in the `vincio.data` subpackage; entirely backward-compatible, dependency-free,
deterministic, and offline. `API_VERSION` is unchanged at `"4.0"` — 4.3 extends the 4.x surface additively (486 → 491
public symbols).

### Added

- **`query_dataset` / `app.query_data` — the governed text-to-query pipeline.** Grounds a natural-language question (or
  explicit SQL, or a dataframe-op pipeline) over a registered `DataCatalog` into a `QueryPlan`: **schema-grounded** (an
  unknown table or column is refused before execution), **read-only-verified** (`is_read_only_sql` / `assert_read_only_sql`
  accept only a single `SELECT` / `WITH` statement with no write / DDL / stacked statement, screened after stripping
  comments and string literals; a breach raises the new `UnsafeQueryError`), dry-run / cost-bounded (compiled and
  plan-inspected without fetching, capped by `max_rows`), and with the natural-language question screened by the same
  injection detector the text rails use. `app.register_dataset` registers a dataset in the app's catalog; both the
  registration (`data_register`) and the query (`data_query`) land on the hash-chained audit log.
- **`InProcessSqlEngine` — the offline, read-only SQL engine.** The default `QueryEngine` executes the verified query on an
  in-memory standard-library `sqlite3` database opened read-only (`PRAGMA query_only`) with an authorizer that **denies
  every non-read action** — defense in depth beneath the read-only screen, so a write or DDL that somehow passed the screen
  is still refused by the engine. Rows go to the engine, not the prompt. A pushdown engine can run the same verified SQL
  against a live source through the `QueryEngine` interface.
- **`QueryResult` — schema-bearing, cell-level cited, offline-verifiable.** Carries the result as a `Dataset`, per-row
  `RowProvenance` of the exact source `CellCitation`s (`cite_refs(row, col)` renders `sales#r0!revenue`), a `LineageCoverage`
  that is `cell` for single-table projection / filter and group-by aggregation (and honestly `result` for shapes outside
  that grammar, never silently downgraded), the content hashes of the source tables, and a `result_hash`. `verify(catalog)`
  re-executes the query and confirms the result — and every cited cell — re-derives from the bytes; a tampered source, a
  tampered result, or a flipped cell flips it to `False`. `to_evidence()` projects the result into cited table evidence.
- **The dataframe-op dialect.** `dialect="dataframe"` runs the same pipeline over the whitelisted, `eval`-free
  `vincio.verify.ProgramOp` transforms (`select` / `filter` / `derive` / `rename`), which are read-only by construction and
  yield **exact per-cell lineage** with no model in the loop — a derived column cites every source column its expression
  references.
- **`make_query_contract` — the read-only guarantee as a `ToolContract`.** Refuses a non-read-only query as a pre-condition
  and bounds the row count as a post-condition, so a `query_data` tool **structurally** refuses a write when it rides the
  permissioned, approval-gated, audited tool runtime.
- **`HeuristicQueryPlanner` — deterministic offline NL→SQL.** Grounds common analyst questions (counts, single-column and
  group-by aggregates) against the catalog schema offline; a question it cannot ground confidently returns `None` rather
  than guessing an ungrounded query.
- **Top-level surface.** `QueryResult`, `QueryPlan`, `DataCatalog`, `CellCitation`, and `query_dataset` are re-exported at
  the package top level; the full surface (`QueryEngine`, `InProcessSqlEngine`, `QueryDialect`, `RowProvenance`,
  `LineageCoverage`, `HeuristicQueryPlanner`, `make_query_contract`, `is_read_only_sql`, `assert_read_only_sql`,
  `QueryError`, `UnsafeQueryError`) lives in `vincio.data`. New error codes `QUERY_ERROR` and `UNSAFE_QUERY`.
- **Spider / BIRD text-to-SQL adapters** (`vincio.evals.SpiderAdapter`, `BIRDAdapter`, with
  `spider_tasks_from_export` / `bird_tasks_from_export` and shipped fixtures) scored by **execution accuracy** — the
  predicted query's result set equals the gold's when both run on the read-only engine, so a generated write is refused and
  scores the task failed. A `text_to_query` section of the **DataPlaneBench** family adds an `execution_accuracy` SLO
  (gated ≥ 0.95), a `read_only_enforced` SLO, and a `provenance_verifiable` SLO.

### Changed

- **`Dataset.from_rows(rows, ["bare", "names"])` now infers column types from the data** (as `from_records` already does),
  instead of defaulting every column to `str`. Bare column names carry no type intent, so a numeric column is now typed
  numeric on the path to the encoder and the query engine; passing typed `ColumnSchema`s is unchanged. This makes
  `app.register_dataset(rows, columns=[...])` and `app.table_evidence(rows, columns=[...])` type their numeric columns
  correctly.

### Documentation

- New concept page **[Governed text-to-query and cell-level provenance](concepts/governed-text-to-query.md)** (under
  `docs/`), a runnable **`examples/15_governed_text_to_query.py`**, and updates to the README, `llms.txt`, `SECURITY.md`,
  the SLO reference, and the ROADMAP (the 4.3 row moves from planned to shipped). The 4.2 profiling concept's "what it is
  not" note is updated, since text-to-query and cell-level provenance now ship.

## [4.2.0] - 2026-06-26

Dataset profiling, sampling & data-quality rails — the second rung of the data & analytics plane: fitting a dataset far
larger than the window into bounded, faithful, screened evidence. Additive in the `vincio.data` subpackage; entirely
backward-compatible (the connector first-N behavior is unchanged unless an opt-in `sample=` is set), dependency-free,
deterministic, and offline. `API_VERSION` is unchanged at `"4.0"` — 4.2 extends the 4.x surface additively (483 → 486
public symbols).

### Added

- **`profile_dataset` / `profile_stream` — deterministic, bounded-memory column profiling.** A `DatasetProfile` carries a
  `ColumnProfile` per column: exact `count` / `null_count` / `null_rate`, exact `min` / `max` / `mean` / `stddev`,
  percentiles and a population-scaled `histogram` (reservoir-estimated and flagged `estimated` once a column exceeds
  `reservoir_size`), `distinct` cardinality (exact up to `distinct_cap`, then a lower bound), `top_values`, and
  `exemplars`. The profile is itself **fixed-size evidence** — `to_evidence_item()` renders it as a compact stats table the
  context compiler scores and cites — and `profile_stream` profiles a row iterator without materializing it.
- **`sample_dataset` (`reservoir_sample` / `stratified_sample` / `systematic_sample`).** A representative sample that stands
  in for the whole, replacing a biased first-N cutoff: uniform reservoir sampling in a single `O(k)`-memory pass,
  stratified sampling that preserves a key column's distribution (proportional, largest-remainder allocation), systematic
  even spacing, and `head`. A sampled `Dataset` records how it was drawn in `metadata["sample"]`. `SampleMethod` enumerates
  the strategies.
- **`fit_to_window` / `fit_stream` (`WindowFit`).** The fit-in-window guarantee: a table far larger than the window is
  represented under a **fixed token budget** — a full-fidelity profile (over every row, in bounded memory) plus a
  representative sample sized to whatever budget the profile leaves — and the representation's size is invariant to the row
  count. `to_evidence_items()` yields the profile and the sample as cited table evidence; `fit_stream` does it in a single
  bounded pass over a source larger than memory.
- **`DataQualityRails` (`ColumnConstraint`, `DataQualityReport`, `DataQualityViolation`).** Deterministic screening of a
  tabular input for schema violations (type, nullability, null rate), constraint breaks (range, allowed-set, pattern,
  uniqueness, monotonicity), and numeric anomalies (robust median/MAD outliers) — and the same PII / secret / injection
  detectors the text rails ride, applied to string cells. `from_schema` / `from_dataset` derive baseline schema-enforcing
  rails. `DataQualityReport.raise_for_status()` raises the new `DataQualityError` (a `DataError` subclass).
- **App surface.** `app.profile_dataset`, `app.sample_dataset`, `app.fit_dataset`, and `app.screen_data` (which lands an
  audited `data_quality` decision on the hash-chained log). `DatasetProfile`, `DataQualityRails`, and `DataQualityReport`
  are re-exported at the top level; the full profiling/sampling/quality surface lives in `vincio.data`.
- **DataPlaneBench VincioBench family** (`families.data_plane`) with a `dataset_fit_in_window` SLO (a 500k-row table fits a
  fixed token budget, size invariant to row count), a `dataset_profile_faithful` SLO, and a `data_quality_rails_complete`
  SLO, plus a `dataset_fit` head-to-head in `benchmarks/competitive.py` against `json.dumps`, the compact all-rows
  encoding, and `pandas.describe`.

### Changed

- **The SQL / BigQuery / Snowflake connectors gain an opt-in reservoir sample.** Passing `sample=N` (with `sample_seed=`)
  draws a representative sample of `N` rows from the entire result set in a single bounded pass instead of truncating at the
  first `max_rows`. The default (no `sample`) is unchanged: the first-N cutoff behaves exactly as before.

### Documentation

- New concept page **[Dataset profiling, sampling, and quality rails](concepts/dataset-profiling.md)** (under `docs/`), a
  runnable **`examples/14_dataset_profiling.py`**, and updates to the README, `llms.txt`, `SECURITY.md`, the SLO reference,
  the benchmark docs, and the ROADMAP (the 4.2 row moves from planned to shipped).

## [4.1.0] - 2026-06-26

Tabular evidence & the compact data encoder — the token-efficiency foundation of the data & analytics plane. Structured
data becomes *first-class, schema-bearing, columnar evidence*, never a row-flattened `Document`. Additive behind a new
`vincio.data` subpackage; entirely backward-compatible (a table dict without a compact encoding keeps its prior behavior),
dependency-free, deterministic, and offline. `API_VERSION` is unchanged at `"4.0"` — 4.1 extends the 4.x surface additively.

### Added

- **`vincio.data` — a typed, columnar `Dataset`.** A `Dataset` carries a typed `DataSchema` (a `ColumnSchema` per column:
  name, `DataType`, optional unit, nullability) over column-major cells, built from rows, records, columns, a legacy
  `TableData`, or a compact encoding (`from_rows` / `from_records` / `from_columns` / `from_table_data` / `from_encoding`),
  with `rows` / `records` / `column` / `head` / `exemplars` / `to_table_data` / `to_evidence` accessors. `DataSchema`,
  `ColumnSchema`, and `DataType` are exported from `vincio.data`.
- **`DataEncoder` — the deterministic, token-oriented encoder.** Renders a dataset header-once — the schema, types, units,
  and null-handling declared once and the cells as delimited rows — in a compact form that is **lossless** (`decode`
  reconstructs the columns, types, and cells from the bytes alone: a null distinguished from an empty string; leading zeros,
  embedded delimiters, quotes, and newlines preserved) and reports the **columnar-accurate** token cost (the count of the
  tokens the model receives). `encode_value` is the token-efficient replacement for `json.dumps`. Re-exported at the top
  level as `vincio.DataEncoder`; the encoding kernel lives in `vincio.core.tabular`.
- **`TableEvidence` (`app.table_evidence`).** Projects a dataset into the `modality="table"` context evidence the compiler
  already scores, deduplicates, budgets, orders, and cites; its scorable text and prompt rendering are the compact encoding,
  and its token cost is columnar-accurate. The context compiler also coerces a bare `Dataset` / `TableEvidence` in its
  `evidence` list. Re-exported at the top level as `vincio.TableEvidence`.
- **`DataError`** (`vincio.data` / `vincio.core.errors`, code `DATA_ERROR`) with an error-catalog entry.
- **CostBench tabular token-efficiency family** (`families.cost.table_encoding`) plus a `table_token_efficiency` SLO
  (≥ 40% fewer tokens than `json.dumps`) and a `table_encoding_lossless` SLO, and a `data_encoding` head-to-head in
  `benchmarks/competitive.py` against `json.dumps`, `pandas.to_markdown`, and a TOON reference encoder.

### Changed

- **The compact encoder replaces the pipe-join / `json.dumps` rendering on the path to the prompt.** `TableData.to_text`
  now renders the compact, lossless encoding (its string cells preserved exactly) instead of a pipe-joined table, and
  `structure_data` encodes nested JSON-like values compactly instead of `json.dumps(indent=2)`.
- **Columnar-accurate table token accounting.** `EvidenceItem.estimated_token_cost` and the context scorer cost a table
  carrying a compact encoding by the tokens the model actually receives, replacing the generic `3·cells` heuristic; a raw
  table dict without an encoding keeps the prior per-cell heuristic unchanged.

### Documentation

- New concept page **[Tabular evidence and the compact data encoder](concepts/tabular-evidence.md)** (under `docs/`), a
  runnable **`examples/13_tabular_evidence.py`**, and updates to the README, `llms.txt`, `SECURITY.md`, the benchmark docs,
  and the ROADMAP (the 4.1 row moves from planned to shipped).

## [4.0.0] - 2026-06-24

Consolidation, hardening & the long-term-support major. 4.0 is the one announced breaking window — and it breaks nothing.
Every release from 1.0 through 3.49 was additive on a frozen public surface, the mechanical deprecation policy was followed
across 40+ themes, and **no public API ever reached its `removed_in` runway**, so the deprecation sweep removes nothing and a
project that tracked 3.x cleanly upgrades with **zero source changes**. This release promotes the contract version, re-freezes
the public surface for the 4.x line behind a mechanical drift gate, and ships the source-codemod machinery and migration guide
that make a major upgrade a one-shot, mechanical operation. After it, the platform enters long-term support: bug-fix, security,
and standards-tracking releases on a stable 4.x surface. Entirely backward-compatible — the 481-name public surface is
unchanged, no dependency is added, and the whole release runs offline and deterministically.

### Changed

- **`API_VERSION` promoted to `"4.0"`** and the package version to `4.0.0`. `API_VERSION` is the public-API contract version
  SemVer is applied against; it bumps only on a major. No symbol was removed or renamed — the surface re-frozen for 4.x is
  byte-for-byte the 3.49 surface.

### Added

- **`vincio migrate <target>` — the source codemod (`vincio.cli.migrate`).** The code-surface analogue of
  `vincio config migrate`: a static, `ast`-based codemod (it parses project source, never imports or runs it) driven by a
  declarative, per-major `RENAMES` rename table. It rewrites only the exact identifier tokens a rename touches — imports,
  attribute access (`vincio.old`), and bound-name uses, honoring `import ... as` aliases — leaving formatting and unrelated
  code intact. Modes: dry-run (default), `--write` (apply in place), `--check` (CI gate, exits non-zero if a migration is
  available), `--json`. The `"4.0"` table is intentionally empty, so it truthfully reports "no source changes required"; it is
  the mechanism through which any future 4.x consolidation or 5.0 removal is delivered.
- **A mechanical public-surface freeze.** `docs/reference/public-surface.txt` pins the exact public surface
  (`sorted(vincio.__all__)`), regenerable with `python -m vincio._apiref --freeze` (`render_frozen_surface` /
  `load_frozen_surface`). A new `test_public_surface_is_frozen` fails the build if the live surface drifts from the manifest,
  so no SemVer-significant change to the surface can land silently.
- **`MIGRATION.md`** — the 3.x → 4.0 upgrade guide: the (empty) rename table, why there are no renames, the codemod usage,
  and the 4.0 long-term-support contract.

### Documentation

- `ROADMAP.md` marks 4.0 shipped and the plan complete; `README.md`, `llms.txt`, `SECURITY.md` (supported versions → 4.x),
  `docs/reference/stability.md` (the 4.0 LTS contract + the codemod), and `docs/reference/cli.md` (the `migrate` command)
  updated.

## [3.49.0] - 2026-06-24

Continuous assurance cases & production certification — the platform-completion capstone. The platform already *produces*
the evidence a production AI system is judged on — eval and regression gates, the governance-invariant verifier, reasoning
certificates and runtime monitors, identity and delegation provenance, the signed audit chain, C2PA media provenance, and
SBOM / SLSA build attestations. This release adds the capstone that **assembles that evidence into one structured,
machine-checkable argument** that the system is fit for purpose, and keeps that argument **continuously valid as the system
changes** — the assurance-case discipline (GSN / CAE) the safety and regulatory frontier demands. With it, every subsystem
composes into one continuously-verified safety argument and the platform is **production-complete**. Entirely additive and
backward-compatible — `API_VERSION` stays `3.0`, the existing surface is unchanged, no new dependency, and the whole theme
runs offline and deterministically.

### Added

- **Structured, content-bound assurance case (`vincio.assurance`).** An `AssuranceCase` (`app.assurance_case`) is an
  argument tree: a top `Claim` (*this app is fit for purpose X under context Y*) decomposed into sub-claims (combined `all`
  or `any`), each leaf discharged by `Evidence` the platform **already emits** — `Evidence.from_gate` (a `CanaryVerdict` /
  eval gate), `from_governance` (a `GovernanceVerifier` report), `from_certificate` (a reasoning `Certificate`),
  `from_audit` (an `AuditLog` segment), `from_identity` (an identity / delegation chain), `from_sbom` (an `AIBOM`), and
  `asserted` (external) — each **bound by hash** and carrying a freshness `horizon_days`. The whole case `verify`s offline
  (a tampered argument tree is caught), and `case.check(as_of=)` re-derives the verdict into an `AssuranceReport` that
  **pinpoints** every claim whose evidence is `missing`, `stale`, or `falsified` — no claim stands on missing or stale
  evidence.
- **Continuous assurance & the regression gate.** The case is re-checked on every change; `assurance_regression_gate(before,
  after)` turns a claim that held before but is no longer discharged into a **build failure**, the same gate machinery that
  blocks a quality regression. Each `Evidence` re-derives its content hash, so a flipped verdict or a tampered support is
  caught from the bytes; the `AssuranceReport` is content-bound, so an edited verdict is caught on re-verification.
- **Incident response & safety-case learning.** A signed `Incident` ties an observed production failure to the sub-claim it
  falsified; `case.learn_from(incident)` adds a remediation sub-claim that **demands fresh evidence** before the case
  re-validates, and `case.discharge(claim_id, *evidence)` closes it once the fix is proven — closing the loop from a
  production incident back into a stronger, re-verified safety argument.
- **Portable certification (`app.certify`).** `app.certify(case, *, residual_risks=None, aibom=True, sign=True,
  record=True)` emits a `CertificationReport` (the case, its discharged evidence verdict, the residual risks, and the build
  provenance — the `vincio` version, a CycloneDX AI-BOM, an SLSA note), signed with the app's identity. Its `verify()`
  recomputes the report hash *and re-runs the case's own check from the bytes*, so a report certifying a case that does not
  hold is caught offline. Every verdict lands on the hash-chained audit log (`assurance_case` / `assurance_certification`).
  A free `certify(...)` function exposes the same without an app. New `AssuranceError` for a tampered or malformed case /
  certification.
- **Benchmark, SLOs, example & docs.** An `assurance` VincioBench family gates assurance soundness (a fully-evidenced case
  holds and verifies offline while a claim resting on missing, stale, or falsified evidence is pinpointed and fails) and
  assurance regression (a previously-discharged claim now falsified fails the build, a signed incident makes the case demand
  a remediation proof, and a certification report verifies from the bytes), held by an assurance-soundness SLO and an
  assurance-regression SLO. New runnable example `examples/93_assurance.py`, an [assurance guide](docs/guides/assurance.md),
  and synchronized README / SECURITY / llms.txt / API reference / ROADMAP.

## [3.48.0] - 2026-06-24

Autonomous skill acquisition & open-ended curriculum. The closed self-improvement loop, RLVR, and the distillation
flywheel make an agent *better at known tasks*; this release adds the apex of that arc — **open-ended capability growth**.
An agent proposes its own tasks at the edge of its competence, distills successful trajectories into a reusable, versioned
skill library, and bootstraps (Voyager / ADAS-shaped) under the *same* no-regression gate a promotion already clears, so
growth is safe and reversible rather than unbounded drift, and the rails + the governance verifier gate every proposed
objective so the autonomy stays inside the guardrails. Entirely additive and backward-compatible — `API_VERSION` stays
`3.0`, the existing surface is unchanged, no new dependency, and the whole theme runs offline and deterministically against
the reference environments.

### Added

- **Reusable, content-addressed skill library (`vincio.cultivate`).** A `LearnedSkill` is a named, typed, tool-using
  procedure distilled from a successful trajectory — an objective, a precondition, an ordered list of `SkillStep`s (each a
  primitive `EnvAction` **or** a call to an existing skill, so skills **compose**), a verifier, and `SkillProvenance`. A
  `LearnedSkillLibrary` is content-addressed: a byte-for-byte-duplicate procedure **deduplicates**, a changed procedure
  under an existing name **versions**, `compose()` flattens a procedure (refusing a cycle or a missing sub-skill), a skill
  is retrieved like memory and tools (`relevant` / `evidence_for`, via a `to_skill()` projection onto the same
  progressive-disclosure path as `vincio.skills.Skill`), and the whole library is offline-verifiable (`verify()` recomputes
  every skill's content hash; `library_hash` binds the active set; a tampered procedure is caught from the bytes). This is
  distinct from `vincio.skills.SkillLibrary`, which holds human-authored `SKILL.md` knowledge.
- **Self-proposed, bounded curriculum.** An `AutoCurriculum` proposes the next task at the **frontier of current
  competence** — solvable by a bounded, library-composing test-time search (`SkillSearch`) but not yet by retrieving an
  existing skill — and **gates every proposed objective before it is attempted**: the instruction is screened by the
  programmable `RailEngine` and the `GovernanceVerifier` must prove the app's controls hold, so an unsafe or out-of-policy
  objective is pinpointed and refused (a `CurriculumProposal` whose content hash catches a refused objective relabelled as
  proposed). `library_capability` measures the fraction of a held-out frontier set the library solves by applying a known
  skill — the monotonic quantity the gate protects.
- **The cultivation loop (`app.cultivate`).** `app.cultivate(curriculum, *, library=None, held_out=None, cycles=3,
  rails=None, governance=None, min_capability_gain=0.0, prune=True, record=True)` runs propose → attempt → verify →
  distill → promote across cycles. A winning, oracle-verified trajectory is distilled into a `LearnedSkill` and **promoted
  only through the same `no_regression_gate`** a prompt or policy deploy clears (capability on the held-out set must not
  fall, surfaced as the same `CanaryVerdict`); a skill that stops paying its way is **demoted, never silently kept**.
  Returns a content-bound `CultivationResult` whose `verify()` re-derives the monotonicity and stay-in-policy verdicts from
  the bytes, with the grown library on `result.library`; every cultivation lands on the hash-chained audit log
  (`skill_cultivation`) and the event bus (`cultivation.completed`). A `Cultivator` and a free `cultivate(...)` function
  expose the same loop without an app. New `CultivationError` for a tampered or malformed skill / curriculum.
- **Benchmark, SLOs, example & docs.** A `skill_acquisition` VincioBench family gates capability monotonicity (a full
  cultivation run ends at least as capable as it began, composes learned skills, demotes dead weight, and catches a
  tampered capability number) and stay-in-policy safety (a rail-blocked or governance-failing objective is refused and
  never attempted), held by a capability-monotonicity SLO and a stay-in-policy safety SLO. New runnable example
  `examples/92_skill_acquisition.py`, a [skill-acquisition guide](docs/guides/skill-acquisition.md), and synchronized
  README / SECURITY / llms.txt / API reference / ROADMAP.

## [3.47.0] - 2026-06-24

Verified reasoning & neuro-symbolic certificates. The platform graded outputs with judges, oracles, and a governance
verifier — but those per-answer signals are *probabilistic*. This release adds the certifiable frontier: for the classes
of question where it is possible, an answer carries a **checkable certificate** a deterministic verifier confirms
independently of the model — the output-side analogue of the governance verifier's machine-checked invariants. It ships
three planes of deterministic, offline verification: proof-carrying answers, runtime verification & shielding, and
verified tool use. Entirely additive and backward-compatible — `API_VERSION` stays `3.0`, the existing surface is
unchanged, the dependency-free kernels are the default (optional SMT / CAS behind `vincio[verify]`), and the whole theme
runs offline and deterministically.

### Added

- **Proof-carrying answers (`vincio.verify`).** `app.verify_reasoning(answer, *, verifiers=None, evidence=None,
  schema=None, constraints=None, facts=None, now=None, regenerate=None, max_cycles=2, raise_on_refute=False)` runs a set
  of deterministic kernels over an answer and returns a `VerifiedAnswer` whose content-bound `Certificate` is `verified`
  only when a kernel **recomputed** a claim and it held, `refuted` when a recomputation disagreed (a proof the answer is
  wrong), or `inapplicable` when no checkable claim of that kind exists. The certificate **re-derives its verdict from the
  bytes** (`certificate.verify()`), so a flipped status is caught. A refuted certificate **refuses to emit**, and a
  `regenerate` callback drives the existing bounded self-correction loop to repair it — the refuse-or-repair discipline
  structured output uses, now over *reasoning*.
- **Deterministic kernels.** A pluggable `ReasoningVerifier` protocol and a `CompositeVerifier` over `ArithmeticVerifier`,
  `UnitVerifier` (refuses a dimensional mismatch), `TemporalVerifier` (real-calendar date math), `ConstraintVerifier`
  (constraint satisfaction over a typed `Constraint`), `SchemaVerifier`, and `CitationVerifier` (strict entailment with
  number checking) — `default_verifiers()`, all pure and offline. A safe `safe_eval_arithmetic` recursive-descent
  evaluator (never `eval`).
- **Runtime verification & shielding.** A `BehaviorSpec` (`forbid` / `require_before` / `invariant`, via `EventPattern`)
  states a temporal-logic-lite property over an agent's trajectory; a `RuntimeMonitor` (`app.behavior_monitor`) checks it
  step-by-step; and a `Shield` (`app.shield(specs, *, mode="block"|"repair"|"monitor", use=False)` / `app.use_shield`)
  **blocks or repairs a violating action *before* it executes**. Wired into the tool runtime (`ToolRuntime(shield=...)`),
  so a policy-violating tool call (an unapproved write) is structurally refused — the per-step, online counterpart of the
  ahead-of-run governance verifier.
- **Verified tool use & synthesized programs.** A `ToolContract` (`ToolClause`, `requires_that` / `ensures_that`) declares
  pre/post-conditions the runtime checks against the actual arguments and result (`app.add_tool(..., contract=...)` /
  `registry.register(..., contract=...)`); a breach raises `ToolContractError`. `synthesize` (`app.synthesize_program`)
  emits a `SynthesizedProgram` from a `ProgramSpec` / `ProgramOp` / `ProgramProperty` whose declared properties are proven
  into a `Certificate` before it runs and re-checked on every use — proof-carrying code in the tool plane.
- **New errors.** `ReasoningVerificationError` (code `REASONING_VERIFICATION_ERROR`) and its subclasses
  `CertificateRefutedError`, `BehaviorViolationError` (code `BEHAVIOR_VIOLATION`), `ProgramSynthesisError` (code
  `PROGRAM_SYNTHESIS_FAILED`); and `ToolContractError` (a `ToolError`, code `TOOL_CONTRACT_VIOLATION`).
- **Optional SMT / CAS backends (`vincio.verify.smt`, extra `vincio[verify]`).** `SmtConstraintVerifier` (Z3, proves a
  constraint system consistent) and `CasArithmeticVerifier` (SymPy, exact rational arithmetic) with `smt_available()` /
  `cas_available()` — strictly opt-in; nothing on the offline path imports them.
- **Benchmark, SLOs, example & docs.** A `verified_reasoning` VincioBench family gates certificate soundness (a wrong
  sum, a dimensional mismatch, an off-by-one duration, a violated constraint, and an uncited number are each refuted; a
  tampered verdict is caught) and shield-prevents-violation (an unapproved write is blocked, an approved one proceeds),
  held by a certificate-soundness SLO and a shield-prevents-violation SLO. New runnable example
  `examples/91_verified_reasoning.py`, a [verified-reasoning guide](docs/guides/verified-reasoning.md), and synchronized
  README / SECURITY / llms.txt / API reference / ROADMAP.

## [3.46.0] - 2026-06-23

Agent identity, delegation & cryptographic accountability. The platform signed contracts, settlements, attestations,
audit entries, and engagement narratives — but *who* a signing key belonged to was an out-of-band assumption (a `key_id`
string a verifier had to trust). This release makes identity itself first-class and verifiable: a portable,
self-certifying agent identity, bounded delegation along a verifiable chain, and key rotation/revocation so a compromised
key cannot forge history. It is the substrate beneath the tool permissions, the agent fabric, and the cross-org trust
fabric — the layer that answers *who authorized this action, down what chain, within what bounds*. Entirely additive and
backward-compatible — `API_VERSION` stays `3.0`, the existing signing surface is unchanged, and the whole theme runs
offline and deterministically (pure-Python RFC 8032 Ed25519 by default; the native `cryptography` backend behind
`vincio[crypto]`).

### Added

- **Portable, self-certifying identity (`vincio.security.identity`).** `app.identity(name=None, *, controller="",
  capabilities=None, seed=None, use=False)` mints an `AgentIdentity` built on an Ed25519 key whose **DID is derived from
  the public key** (`did:vincio:ed25519:<hex>`), so the verifying key resolves from the identifier alone, offline, with no
  registry (`public_key_from_did` / `did_from_public_key` / `is_vincio_did` / `key_fingerprint`). Its content-bound
  `IdentityDocument` (keys, advertised capabilities, rotation history) `verify()`s from the bytes, and an `AgentIdentity`
  satisfies the `ChainSigner` protocol (`key_id` is the DID) so it drops into every signing slot the platform exposes.
- **Key rotation & revocation (`Keyring` / `KeyRecord`).** A `Keyring` rotates keys along a **signed rotation chain**
  (each new key authorized by the one before it), so a signature is validated against the key current *at signing time*:
  `document.verify_signature(message, signature, at=...)` reports the signing key and whether it was active then. A
  rotated-away or revoked key cannot sign new history, while signatures it made while current stay valid — modelling key
  compromise without invalidating legitimate past acts.
- **Delegation chains & attenuated authority (`Delegation` / `DelegationChain` / `Grant`).** A signed `Delegation` mints a
  bounded `Grant` (a subset of capabilities, a budget cap, an expiry, an audience, a re-delegation depth) from a principal
  to an agent; `delegation.delegate(...)` sub-delegates to a sub-agent, composing into a `DelegationChain` that
  `verify(root_issuer=...)`s **offline** under one invariant — **each link only attenuates its parent's grant, never
  amplifies it**. `chain.permits(capability, budget_usd=, at=, audience=)` / `require_permits(...)` answer authorization;
  an over-reaching sub-delegation or a tampered grant is **refused from the bytes**. A rotated-key link carries a compact
  `KeyAuthorization` so the chain stays registry-free.
- **Verifiable credentials (`AgentCredential`).** `app.issue_credential(subject, claims, *, as_identity=None,
  not_after=None, expires_in=None)` issues a signed claim (*admitted to capability X*, *operated by org Y*) an importer
  `verify()`s offline and folds into the capability-gated admission path (`credential.admits(capability)`); a tampered
  claim or a forged issuer is caught from the bytes.
- **Accountable audit binding.** `app.use_identity(identity)` binds an identity as the content / contract / audit signer,
  so subsequent audit entries, negotiated contracts, and settlement records record the identity's **DID** as their
  `key_id` — accountability as a cryptographic fact, not a logged string.
- **New error.** `IdentityError` (a `SecurityError`, code `IDENTITY_VERIFICATION_FAILED`) is raised when a DID is
  malformed, a document or rotation chain does not verify, a sub-delegation amplifies its parent, or a credential's
  signature does not bind to its issuer DID.
- **Crypto backend (`vincio.security._ed25519`, extra `vincio[crypto]`).** A pure-Python, RFC 8032-conformant Ed25519
  (the dependency-free default) with an automatic switch to the native, constant-time `cryptography` backend when the
  `crypto` extra is installed — byte-identical signatures, so artifacts interoperate across backends.
- **Benchmark, SLOs, example & docs.** An `identity` VincioBench family gates identity integrity (self-certifying DID,
  document verification, signed rotation chain, old-signature survival) and delegation attenuation (offline chain
  verification, refusal of amplification and tampering), held by an identity-integrity SLO and a delegation-attenuation
  SLO. New runnable example `examples/90_agent_identity_delegation.py`, an [identity guide](docs/guides/agent-identity.md),
  and synchronized README / SECURITY / llms.txt / API reference / ROADMAP.

## [3.45.0] - 2026-06-23

Computer-use & embodied action plane. Computer-use and provider-hosted tools already shipped as a *capability* — a flat
navigate / click / type / screenshot vocabulary behind a pluggable `IsolationBackend` — but as a thin GUI adapter, not a
first-class, grounded, verified, reversible **action plane**. This release adds the rung above it: an agent that drives a
screen *safely* by closing a perceive → ground → pre-gate → act → post-verify → undo-on-divergence loop under the same
governance, budget, rails, and audit the rest of the platform enforces. Entirely additive and backward-compatible —
`API_VERSION` stays `3.0`, the flat `enable_computer_use` / `use_hosted_tools` surface is unchanged, and the whole theme
runs offline and deterministically.

### Added

- **Grounded perception & action (`vincio.tools.computer_environment`).** `app.computer_use(backend="mock", *, screen=,
  policy=, approve=, auto_undo=True, max_steps=50, require_isolation=False)` returns a `ComputerEnvironment` over a
  pluggable `ScreenBackend`. A backend turns a screenshot plus an accessibility tree into a typed `ScreenState` of
  addressable `UIElement`s, and an intent into a typed `UIAction` (`navigate` / `click` / `type` / `scroll` / `drag` /
  `key` / `wait`) bound to a target by a **stable selector** (role + accessible name), not a pixel — so an action is
  replayable, auditable, and survives a layout shift. The deterministic `MockScreen` drives an in-process `ScreenApp` /
  `ScreenSpec` offline; `PlaywrightScreen` (browser / CDP), `AccessibilityScreen` (an OS accessibility tree), and
  `RemoteDesktopScreen` (a remote machine) ride a real driver behind `vincio[computer-use]`.
- **Pre-gated, post-verified, reversible steps.** Every `env.act(action)` runs through an `ActionPolicy` pre-gate — a
  destructive or out-of-scope action is gated like a write tool, behind an `approve` callback — then performs the effect,
  **post-verifies** it against the action's declared expectation (a declarative `StateCheck` / `expect_change`), and on
  divergence **undoes** it (a synthesized inverse, falling back to a prior-state restore) into a typed `ActionOutcome`
  (`ActionDecision`). Every gate decision, action, divergence, and undo lands on the hash-chained audit log
  (`computer_use_session` / `computer_action`).
- **Task grounding & trajectory.** A `ComputerTask` carries a goal and a declarative verifier; `env.run(policy, task)` /
  `env.arun` drive a policy to a verified end state and return a `ComputerRun` (`.success` from the end-state oracle,
  `.safe` = no destructive action ran without approval, `.trajectory`) — projecting onto the same
  `vincio.evals.trajectory.Trajectory` the existing trajectory metrics, test-time search, and world-model planner score,
  with no new search machinery. `make_web_checkout()` is a deterministic, WebArena/OSWorld-shaped reference app.
- **New error.** `ComputerUseError` (`COMPUTER_USE_ERROR`, a `ToolError`) for an undriveable backend, a missing optional
  driver, an unaddressable target, or an exhausted action budget — with a catalog entry and remediation.
- **Benchmark, SLOs, example & docs.** A `computer_use` VincioBench family gates success-at-budget, grounded-stable-selector,
  continuous audit, destructive-gating, no-unapproved-destructive, out-of-scope-gating, and undo-on-divergence — held by a
  `computer_use_success_at_budget` SLO and a `computer_use_no_unapproved_destructive` safety SLO.
  `examples/89_computer_use_action_plane.py` drives the reference app to a verified goal; a new
  [computer-use guide](docs/guides/computer-use.md), the [SLO reference](docs/reference/slo.md), README, ROADMAP, SECURITY,
  `llms.txt`, and the agents concept doc all carry the action plane. The new public symbols (`ComputerEnvironment`,
  `ComputerTask`, `ComputerRun`, `UIElement`, `ScreenState`, `UIAction`, `ActionOutcome`, `ActionPolicy`, `MockScreen`,
  `ScreenApp`, `make_web_checkout`) are exported from `vincio` and `vincio.tools`.

## [3.44.0] - 2026-06-23

Cross-org settlement fabric — unification, conformance & closure. Twenty consecutive minors (3.24–3.43) built the
cross-org settlement & credit **primitives** rung by rung — choreography, settlement, netting, arbitration, portable
reputation, admission, collateral escrow / pooling / rehypothecation guards, proof-of-reserves and proof-of-solvency,
liability completeness / non-equivocation / history, and insolvency resolution by seniority waterfall with close-out
set-off — each signed, content-bound, and offline-verifiable on its own. This release is the **capstone**: it composes the
whole fabric into one lifecycle object, proves it conforms as a system, and **freezes** the domain. Entirely additive and
backward-compatible — `API_VERSION` stays `3.0`, every existing entry point is unchanged and still usable directly, and
the whole theme runs offline and deterministically.

### Added

- **Unified engagement lifecycle (`vincio.settlement.engagement`).** `app.cross_org_engagement(*, buyer="", seller="",
  scope="", coordinator=None)` returns a `CrossOrgEngagement` — a purely-compositional facade that threads the entire
  pipeline behind one governed, audited call-path: `negotiate` → `admit` → `choreograph` (pass `directory=` to discover) →
  `settle` / `settle_saga` → `net` → `arbitrate` → `attest_reputation` / `import_reputation` → `post_escrow` /
  `post_collateral_pool` / `guard_collateral` → `attest_custody` / `attest_liabilities` / `prove_solvency` /
  `check_completeness` / `check_root_consistency` / `check_history_consistency` → `resolve_insolvency`. Each lifecycle
  method delegates to the *same* `app.*` primitive a caller would use directly (each unchanged), captures the artifact
  (exposed as `eng.contract` / `.delivery` / `.netting` / `.insolvency` / …), and records it as a stage; `record_stage`
  is an escape hatch for any other artifact. The facade adds no new economic logic.
- **Content-bound engagement narrative.** `eng.seal(*, sign=True, record_audit=True)` mints an `EngagementNarrative`: an
  ordered chain of `EngagementStage`s, each binding the lifecycle verb, the captured artifact's own content hash, and a
  digest of its bytes into a hash-chained link, sealed into one content hash the coordinator signs and landed on the audit
  log (action `cross_org_engagement`). `EngagementNarrative.verify(verifier=None, *, require=None, artifacts=None)`
  recomputes the whole chain from the bytes alone (`intact` / `head_ok` / `hash_ok` / `digests_ok` / `signatures_ok` /
  `broken_at`), and `eng.verify(verifier)` re-digests every captured live artifact against its bound digest — so a
  re-ordered stage, an edited digest, a broken link, an edited underlying artifact, or a forged signature is caught.
  `require_valid`, `stage_names`, `stage`, `to_wire` / `from_wire`, `print_summary`. `CrossOrgEngagement`,
  `EngagementNarrative`, `EngagementStage`, `EngagementSignature`, `EngagementVerification` are public.
- **End-to-end conformance bench.** A `cross_org_conformance` VincioBench family drives a complete engagement and gates
  that the lifecycle threads every stage, the narrative chains and verifies offline, every captured artifact verifies, one
  continuous hash-chained audit narrative recomputes, a tamper introduced anywhere is caught, and the facade is purely
  compositional — held by a `cross_org_conformance_end_to_end` SLO and a `cross_org_conformance_tamper_evident` SLO.
- **Example & docs.** `examples/88_cross_org_engagement_lifecycle.py` drives a full multi-org engagement to a verified
  close; the [settlement guide](docs/guides/settlement.md), [SLO reference](docs/reference/slo.md), README, ROADMAP,
  SECURITY, and `llms.txt` all carry the capstone, and declare the cross-org settlement & credit surface
  **feature-complete and frozen** — no further cross-org *primitive* is scheduled.

## [3.43.0] - 2026-06-23

Cross-org insolvency set-off & close-out netting. The insolvency waterfall (3.42) distributes a poster's reserves across
the creditors it owes — but a creditor is often **also** a debtor of the same counterparty across a web of contracts, and
the waterfall pays it on its **gross** claim while it still owes the estate the other side. Real insolvency law resolves
this first with **set-off** (close-out netting): mutual obligations collapse to a single net claim before any
distribution. The fabric already nets bilateral *settlements* multilaterally (`net_settlements`); this release applies the
same to the liability side *before* the waterfall. Entirely additive and backward-compatible — `API_VERSION` stays `3.0`,
a resolution with no set-off hashes exactly as before, every existing custody, solvency, completeness, non-equivocation,
history, seniority-waterfall, escrow, pooling, rehypothecation, admission, and settlement path is unchanged, and the whole
theme runs offline and deterministically.

### Added

- **Signed set-off statement (`vincio.settlement.setoff`).** `build_set_off_statement(poster, creditor, owed_usd,
  owing_usd, *, references=None, as_of=None)` states the obligations running both ways between a poster and one creditor
  — what the poster owes and what the creditor owes back — into a sealed, unsigned `SetOffStatement` collapsed to the
  poster's bounded net liability (`max(0, owed − owing)`). Signed by **both** parties (a mutually-agreed close-out);
  `verify(verifier=None, *, require=None, require_mutual=False)` recomputes the hash, re-derives the net from the two
  gross figures (an over-stated set-off or a tampered net is caught), and refuses a one-sided claim with `require_mutual`.
  `poster_net_claim_usd` / `set_off_usd` / `creditor_in_debit` / `eliminated` / `direction` / `mutual`, `require_valid`,
  `to_wire` / `from_wire`. `set_off_from_records(poster, creditor, liabilities, records, *, as_of=None, verifier=None)`
  derives a statement straight from the existing signed `LiabilityAttestation` (the `owed_usd`) and `SettlementRecord`s
  (the `owing_usd`, deduped by reconciliation hash) — a tampered artifact refused, a forged signature too with a verifier.
  `SetOffStatement`, `SetOffVerification`, `build_set_off_statement`, `set_off_from_records` are public.
- **Close-out netting into the waterfall.** `resolve_insolvency(..., set_off=[...])` reduces each creditor's proven
  liability to its **net** claim before distributing the reserves — a creditor in debit recovers nothing, and the
  distributable estate shrinks to the true net exposure. Applied after `completeness` (so it nets the *completed* gross),
  reconciled against that gross (an over-stated set-off claiming a different gross, a one-sided statement, a wrong-poster
  statement, or a creditor set off twice is refused), and bound into the resolution by hash. `InsolvencyResolution` gains
  `gross_liabilities_usd` / `set_off_usd` / `set_off_hashes` and `set_off`; `CreditorRecovery` gains `gross_claim_usd` /
  `set_off_usd` / `set_off`; `verify(verifier=None, schedule=None, set_off=None, *, require=None)` re-derives every net
  claim from the recorded gross and the applied set-off (an inflated set-off caught even after re-sealing) and binds the
  mutually-signed statements (`set_off_bound`). All fields default such that a resolution with no set-off is byte-identical
  to before.
- **App & book methods.** `app.build_set_off_statement` / `book.build_set_off_statement` sign and record the statement
  (action `liability_set_off`, decision `poster_owes` / `creditor_in_debit` / `eliminated`); `book.build_set_off_statement`
  can derive it from the book's own settlement records via `liabilities=`. `app.resolve_insolvency` /
  `book.resolve_insolvency` gain a `set_off=` parameter.
- **VincioBench, SLO, example, docs.** The `reputation_portability` family gains `set_off_nets_before_waterfall` and
  `set_off_auditable_offline` with a published insolvency-set-off SLO and CI budgets; example
  `87_cross_org_insolvency_set_off.py`; the settlement guide, README, llms.txt, SECURITY.md, ROADMAP, and the generated
  API index are updated. The public surface grows from 392 to 396 symbols.

## [3.42.0] - 2026-06-23

Cross-org insolvency resolution & liability seniority waterfall. A `SolvencyProof` (3.38) *flags* an insolvency when a
counterparty's proven liabilities exceed its proven reserves, and a history walk (3.41) pinpoints a debt that silently
vanished — but when the reserves genuinely **cannot** cover every obligation, nothing yet said **which** creditors the
available capital pays, and in what order. An insolvency was flagged, not resolved; every creditor was left to assume it
was made whole. The rehypothecation guard already apportions a scarce stake across beneficiaries pari-passu; the liability
side needed the same, plus the **seniority** real obligations carry. This release ranks the obligations into a signed
priority schedule and distributes the proven reserves across them by seniority then pari-passu within a tranche, resolving
an insolvency into who-gets-what. Entirely additive and backward-compatible — `API_VERSION` stays `3.0`, every existing
custody, solvency, completeness, non-equivocation, history, escrow, pooling, rehypothecation, admission, and settlement
path is unchanged, and the whole theme runs offline and deterministically.

### Added

- **Signed seniority schedule (`vincio.settlement.waterfall`).** `build_seniority_schedule(poster, tranches, *,
  as_of=None)` ranks a poster's obligations into a sealed, unsigned `SenioritySchedule` of priority tranches (rank `0`
  most senior) — `tranches` is an ordered spec: a list of creditor-name lists where position is priority, `SeniorityTranche`
  items, or `{rank, creditors, label}` dicts. Signed by the poster or its creditors; `verify` recomputes the hash and
  refuses a malformed ranking (a duplicate rank, a creditor in two tranches). An unlisted creditor falls to the
  most-junior `residual_rank`. `SeniorityTranche`, `SeniorityVerification`, `SenioritySchedule`,
  `build_seniority_schedule` are public.
- **Insolvency waterfall.** `resolve_insolvency(custody, liabilities, schedule=None, *, poster=None, completeness=None,
  solvency=None, as_of=None, verifier=None)` reuses `prove_solvency` (so a tampered/forged/wrong-poster attestation is
  refused) and distributes the proven reserves across the obligations **by seniority then pari-passu within a tranche**
  into an `InsolvencyResolution` — a senior tranche paid in full before a junior one, a partly-funded tranche split
  proportionally — pinpointing each creditor's bounded `CreditorRecovery` (recovery + shortfall) and per-rank
  `WaterfallTranche`. With no schedule the whole set is one pari-passu tranche; pass `completeness` to distribute against
  the completed liability set. `verify(verifier=None, schedule=None, *, require=None)` re-derives the entire distribution
  from the recorded claims/ranks/reserves (an over-stated recovery, a re-ordered tranche, or a junior creditor paid ahead
  of a senior one is refused even after re-sealing) and binds the schedule by hash. `solvent` / `insolvent` /
  `fully_recovered` / `shortfall_bearers` / `recovery_of`, `require_valid` / `require_fully_recovered`.
  `CreditorRecovery`, `WaterfallTranche`, `InsolvencyResolution`, `InsolvencyResolutionVerification`, `resolve_insolvency`
  are public.
- **App & book methods.** `app.build_seniority_schedule` / `book.build_seniority_schedule` sign and record the schedule
  (action `seniority_schedule`, decision `self_ranked` / `ranked`); `app.resolve_insolvency` / `book.resolve_insolvency`
  sign and record the resolution (action `insolvency_resolution`, decision `solvent` / `resolved`) and credit a failure
  against a poster that could not make its creditors whole on the bound `ReputationLedger`.
- **VincioBench, SLO, example, docs.** The `reputation_portability` family gains `insolvency_resolution_distributes` and
  `insolvency_resolution_auditable_offline` with a published insolvency-resolution SLO and CI budgets; example
  `86_cross_org_insolvency_resolution.py`; the settlement guide, README, llms.txt, SECURITY.md, ROADMAP, and the
  generated API index are updated. The public surface grows from 383 to 392 symbols.

## [3.41.0] - 2026-06-23

Cross-org liability history consistency & snapshot monotonicity. Non-equivocation (3.40) catches a counterparty signing
**different** liability roots for the **same** instant, but it is scoped to one `as_of`: a counterparty can still issue a
*later* snapshot that quietly **drops** a past obligation — a debt committed at `T` simply absent from the root it signs
at `T'` — each snapshot internally sound, nothing tying one attestation to its predecessor. Equivocation is conflict
*across creditors*; this is consistency *across time*. This release links a liability snapshot to its predecessor's root
and walks the chain, pinpointing any debt that vanished without a signed, creditor-issued discharge. Entirely additive
and backward-compatible — `API_VERSION` stays `3.0`, the predecessor commitment is bound into the signed hash only when
present (so a standalone attestation hashes exactly as before), every existing custody, solvency, completeness,
non-equivocation, escrow, pooling, rehypothecation, admission, and settlement path is unchanged, and the whole theme runs
offline and deterministically.

### Added

- **Linked liability history (`vincio.settlement.solvency`).** `LiabilityAttestation` carries an optional
  `prior_hash` / `prior_root` / `prior_as_of` commitment to the preceding snapshot (`has_prior`), bound into the signed
  content hash **only when present**. `attest_liabilities(poster, liabilities, *, attestor=None, as_of=None, prior=None)`
  and `LiabilityAttestation.link_to(prior)` set it, requiring the same `(poster, attestor)` and a strictly later `as_of` —
  a back-dated or cross-counterparty link is refused, and a back-dated link is caught from the bytes (`verify`).
- **Discharge.** `discharge_liability(poster, creditor, amount_usd, *, as_of=None, note="")` builds a `Discharge` — the
  **creditor's** signed, content-bound release of part of what a poster owes it (only the creditor signs; a poster cannot
  forge its own). `verify(verifier=None, *, require=None)` checks the hash and signature. `Discharge`,
  `DischargeVerification`, `discharge_liability` are public.
- **check_history_consistency.** `check_history_consistency(attestations, *, discharges=None, verifier=None)` groups
  snapshots by `(poster, attestor)`, walks each chain in `as_of` order, and folds it into a `HistoryConsistencyProof`
  (embedding the whole snapshots and the discharges that explained a drop). A creditor obligation that shrinks between
  snapshots without a signed, in-window discharge is a pinpointed `MonotonicityBreach`; `verify` re-derives every breach
  from the bytes (a dropped breach, a forged or poster-signed discharge, or a back-dated snapshot is caught), and a
  tampered or unsigned snapshot is excluded as inadmissible. `monotone` / `consistent` / `chain_linked`,
  `require_monotone()` / `require_linked()`. Returns a `HistoryConsistencyReport` (`consistent` / `checked` / `chains` /
  `proofs` / `breaching_posters`, `require_consistent()`). `MonotonicityBreach`, `HistoryConsistencyProof`,
  `HistoryConsistencyProofVerification`, `HistoryConsistencyReport`, `check_history_consistency` are public.
- **App & book methods.** `app.check_history_consistency` / `book.check_history_consistency` record each inconsistent
  history on the audit chain (action `liability_history`, decision `consistent` / `inconsistent`) and credit a failure
  against the breaching poster on the bound `ReputationLedger`; `app.discharge_liability` / `book.discharge_liability`
  sign and record a discharge (action `liability_discharge`). `attest_liabilities` on both gains `prior=`.
- **VincioBench, SLO, example, docs.** The `reputation_portability` family gains `history_detects_silent_drop` and
  `history_auditable_offline` with a published liability-history-consistency SLO and CI budgets; example
  `85_cross_org_liability_history_consistency.py`; the settlement guide, README, llms.txt, SECURITY.md, ROADMAP, and the
  generated API index are updated. The public surface grows from 375 to 383 symbols.

## [3.40.0] - 2026-06-23

Cross-org liability non-equivocation & root consistency. Completeness (3.39) proves each creditor's claim is *included*
in the attested liabilities, but a counterparty issues its liability attestation **per relationship**: nothing yet
stopped it presenting **different liability roots** — a *smaller* total — to different creditors, so each creditor's
inclusion proof verifies against the root *it* was shown while the counterparty equivocates across the set. Completeness
catches an omission only when the *omitted* creditor folds its own claim; equivocation hides the omission by showing each
creditor a root on which its own claim *is* present. This release lets creditors compare the signed roots and folds two
conflicting roots one poster signed for the same instant into a non-repudiable equivocation proof. Entirely additive and
backward-compatible — `API_VERSION` stays `3.0`, every existing custody, solvency, completeness, escrow, pooling,
rehypothecation, admission, and settlement path is unchanged, and the whole theme runs offline and deterministically.

### Added

- **RootCommitment (`vincio.settlement.solvency`).** `LiabilityAttestation.root_commitment()` produces a signed,
  privacy-preserving `RootCommitment` — the `liabilities_root` and `as_of` the attestor signed, carried with the
  attestor's signature over the content hash but **without** the line items — that creditors compare over the existing
  attestation exchange. `consistency_key` is `(poster, attestor, as_of)`; `conflicts_with(other)` detects two roots a
  poster signed for the same key; `verify(verifier=None)` checks the embedded attestor signature, refusing a forged one.
  `RootCommitment`, `RootCommitmentVerification` are public.
- **EquivocationProof.** `prove_equivocation(first, second, *, verifier=None, first_creditor="", second_creditor="")`
  folds two attestations a poster signed for the same `(poster, attestor, as_of)` with **different** roots into a
  content-bound `EquivocationProof` that embeds both whole attestations (in canonical content-hash order) and names the
  poster, the two roots, and the creditor each was shown. `verify` re-derives each embedded root from the bytes (a
  mislabeled root cannot survive) and, with the verifier, checks the attestor signed each — so a forged conflicting root
  is refused. It refuses different posters/instants (distinct snapshots) or identical roots (no conflict). `EquivocationProof`,
  `EquivocationProofVerification` are public.
- **check_root_consistency.** `check_root_consistency(attestations, *, verifier=None)` groups a set of held attestations
  (bare or `(creditor, attestation)` pairs) by their `(poster, attestor, as_of)` key and folds every conflicting pair
  into a `RootConsistencyReport` (`consistent` / `checked` / `keys` / `equivocations` / `equivocating_posters`,
  `require_consistent()`). A tampered or — with a verifier — forged/unsigned root is excluded as inadmissible evidence,
  so it cannot manufacture a false accusation. `RootConsistencyReport`, `check_root_consistency` are public.
- **App & book methods.** `app.check_root_consistency` / `book.check_root_consistency` record each equivocation on the
  audit chain (action `liability_equivocation`, decision `equivocation`) and credit a failure against the equivocating
  poster on the bound `ReputationLedger`.
- **VincioBench, SLO, example, docs.** The `reputation_portability` family gains `equivocation_detects_conflicting_roots`
  and `equivocation_auditable_offline` with a published liability-non-equivocation SLO and CI budgets; example
  `84_cross_org_liability_non_equivocation.py`; the settlement guide, README, llms.txt, SECURITY.md, ROADMAP, and the
  generated API index are updated. The public surface grows from 367 to 374 symbols.

## [3.39.0] - 2026-06-23

Cross-org liability inclusion proofs & completeness. Proof-of-solvency (3.38) folds a proven liability *total* against
the proven reserves, but that total is still the attestor's single number: a counterparty could **under-state** what it
owes by quietly omitting a creditor and still attest a sound, re-deriving total over the creditors it *did* list — the
canonical second half of a proof-of-liabilities, where each creditor proves its own claim is **included** so the total
is provably **complete**, not merely internally consistent. This release commits the liability line items into a Merkle
root, gives each creditor an offline-verifiable inclusion proof, and folds a completeness check into the proof-of-solvency
path so the solvency margin is bounded by the obligations creditors can prove. Entirely additive and backward-compatible
— `API_VERSION` stays `3.0`, the existing custody, solvency, escrow, pooling, rehypothecation, admission, and settlement
paths are unchanged (`prove_solvency` without `completeness=` behaves exactly as before), and the whole theme runs offline
and deterministically.

### Added

- **Merkle commitment over liability line items (`vincio.settlement.solvency`).** `LiabilityAttestation` now commits its
  line items into a `liabilities_root` bound into the signed `content_hash`. The total *and* the root re-derive from the
  line items on every verify, so a tampered, dropped, or reordered line is caught even after re-sealing. Leaf and interior
  hashes are domain-separated (the Merkle second-preimage guard) and each leaf binds the creditor's sorted position.
- **InclusionProof.** `LiabilityAttestation.inclusion_proof(creditor)` (and `inclusion_proofs()` for all lines) builds an
  offline-verifiable `InclusionProof` that a creditor's claim is a leaf of the attested root. `verify(attestation=None,
  verifier=None)` reconstructs the root from the leaf and authentication path (`MerkleStep` list) and, against the
  attestation, checks the cited root and leaf belong to the signed attestation — refusing a tampered leaf, a forged root,
  or a root lifted from a different attestation. `MerkleStep`, `InclusionProof`, `InclusionProofVerification` are public.
- **CompletenessProof.** `check_completeness(liabilities, claims, *, verifier=None, as_of=None)` folds creditor claims (a
  `{creditor: amount}` mapping, or `LiabilityLine` / `SettlementRecord` / `(creditor, amount)` items) against a
  `LiabilityAttestation` into a sealed, content-bound `CompletenessProof`, pinpointing every omitted or under-stated claim
  as an `OmissionBreach` and raising the attested figure to a `completed_usd` total. It refuses a tampered attestation (a
  forged attestor too, with the verifier); `verify` re-derives the completed total and the breaches (a dropped omission
  caught by `completed ≥ claimed`, a hidden breach caught on re-seal). `OmissionBreach`, `CompletenessProof`,
  `CompletenessVerification`, `check_completeness` are public.
- **Proof-of-solvency wiring.** `prove_solvency(custody, liabilities, *, poster=None, completeness=None, as_of=None,
  verifier=None)` reads a folded `CompletenessProof`'s completed total instead of the attestor's figure (refusing a check
  for a different poster or attestation), bounding the solvency margin by the obligations creditors can prove. `SolvencyProof`
  gains `attested_liabilities_usd`, `completeness_hash`, `completeness_adjusted`, and `understated_usd`.
- **App & book methods.** `app.inclusion_proof` / `book.inclusion_proof`, `app.check_completeness` / `book.check_completeness`
  (the book derives claims from its owner's own settled records against the poster when `claims` is omitted, via
  `book.claims_against(poster)`), and a `completeness=` parameter on `app.prove_solvency` / `book.prove_solvency`. The
  completeness check is signed and recorded on the audit chain (action `liability_completeness`, decision
  `complete` / `incomplete`).
- **VincioBench, SLO, example, docs.** The `reputation_portability` family gains `inclusion_proof_detects_omission`,
  `completeness_bounds_solvency`, and `completeness_auditable_offline` with a published liability-completeness SLO and CI
  budgets; example `83_cross_org_liability_completeness.py`; the settlement guide, README, llms.txt, SECURITY.md, and the
  generated API index are updated. The public surface grows from 360 to 367 symbols.

## [3.38.0] - 2026-06-23

Cross-org custody liability attestation & proof-of-solvency. Proof-of-reserves (3.37) proves the capital a
counterparty *holds*, so the rehypothecation guard bounds its pledges against a verified figure. But reserves are
only one side of the ledger: a counterparty solvent against one buyer's pledges may be deeply **under-water** once
*every* obligation it owes is counted — it could prove the same reserves against many buyers while quietly insolvent
across all of them, the canonical gap the proof-of-reserves literature closes next with a **proof-of-solvency**
(`reserves ≥ total liabilities`). This release makes the *liability* side evidence-backed too and folds the two
proofs into a bounded, offline-verifiable solvency margin the guard bounds pledges against. Entirely additive and
backward-compatible — `API_VERSION` stays `3.0`, the existing custody, escrow, pooling, rehypothecation, admission,
and settlement paths are unchanged (the `held=` and `custody=` inputs still work exactly as before), and the whole
theme runs offline and deterministically.

### Added

- **LiabilityAttestation (`vincio.settlement.solvency`).** `attest_liabilities(poster, liabilities, *, attestor=None,
  as_of=None)` issues a sealed, unsigned `LiabilityAttestation` over the total obligations a poster owes — the
  liability analogue of `attest_custody`. `liabilities` is a number, a `{creditor: amount}` mapping, or
  `LiabilityLine` / `(creditor, amount)` items (a negative obligation is refused); the attested `liabilities_usd` is
  their sum, re-derived on every verify. `attestor` defaults to the poster (self-attested when `attestor == poster`);
  otherwise an independent auditor/custodian vouches. Only the attestor signs (`sign(signer, *, party=None)`).
- **Proof-of-solvency.** `prove_solvency(custody, liabilities, *, poster=None, as_of=None, verifier=None)` folds a
  proven `CustodyAttestation` against a proven `LiabilityAttestation` for the same poster into a sealed, unsigned
  `SolvencyProof` — a bounded solvency `margin_usd` (`reserves − liabilities`). It **refuses** a tampered figure, a
  forged issuer (with `verifier`), or a custody / liability pair for different posters. The proof exposes
  `solvent` / `insolvent` / `status`, `solvency_adjusted_held` (`max(0, margin)` — the unencumbered capital), and an
  `InsolvencyBreach` (`.poster` / `.custodian` / `.attestor` / `.custody_hash` / `.liability_hash` / `.reserves_usd` /
  `.liabilities_usd` / `.shortfall_usd`) when the liabilities exceed the reserves, with `require_solvent()` raising on
  it.
- **Read by the guard.** `guard_collateral(pools, *, poster=None, held=None, custody=None, solvency=None,
  verify_with=None)` reads a `solvency` proof's `solvency_adjusted_held` as the `held` figure (`solvency_adjusted` /
  `attestor` / `liability_hash` / `liabilities_usd` / `solvency_margin_usd` / `gross_reserves_usd` / `insolvent` /
  `require_solvent()` on the `CollateralLedger`), bounding pledges against capital **not already owed elsewhere**.
  `held=` / `custody=` / `solvency=` are mutually exclusive — the held figure has one source. A tampered proof, a
  forged signature (with `verify_with`), or a proof for a different poster is refused.
- **Offline-verifiable & refused-on-tamper.** `LiabilityAttestation.verify(...)` →
  `LiabilityAttestationVerification(valid, hash_ok, liabilities_sound, signatures_ok, signed_by, reason)` re-derives
  the liability total from the line items; `SolvencyProof.verify(...)` →
  `SolvencyProofVerification(valid, hash_ok, margin_sound, signatures_ok, signed_by, reason)` re-derives the margin
  and the insolvency breach — so a tampered figure or a flipped solvency verdict is caught even after re-sealing.
  `to_wire` / `from_wire` round-trip preserves verification.
- **App / book surface.** `app.attest_liabilities(...)` / `book.attest_liabilities(...)` sign the attestation as the
  attestor and record the issuance on the hash-chained audit log (action `liability_attestation`, decision =
  `self_attested` / `attested`); `app.prove_solvency(...)` / `book.prove_solvency(...)` sign and record the proof
  (action `solvency_proof`, decision = `solvent` / `insolvent`); `app.guard_collateral(..., solvency=)` /
  `book.guard_collateral(..., solvency=)` pass it through.
- **Public surface.** `LiabilityAttestation`, `LiabilityAttestationVerification`, `LiabilityLine`, `InsolvencyBreach`,
  `SolvencyProof`, `SolvencyProofVerification`, `attest_liabilities`, and `prove_solvency` are exported from `vincio`
  and `vincio.settlement`.

### Benchmarks & SLOs

- **VincioBench `reputation_portability`** gains `solvency_bounds_held`, `insolvency_pinpoints`, and
  `solvency_auditable_offline`, gated by a new **proof-of-solvency** SLO (a liability attestation folds against the
  reserve proof into a solvency-adjusted held figure, a proven insolvency is pinpointed as an insolvency breach with
  zero free capital, and a tampered / forged / wrong-poster attestation is refused while the proof lands on the audit
  chain).

### Docs

- New runnable example `82_cross_org_proof_of_solvency.py`; settlement guide, README, llms.txt, SECURITY, and ROADMAP
  updated; the next scheduled theme is **Cross-org liability inclusion proofs & completeness** (target 3.39).

## [3.37.0] - 2026-06-23

Cross-org collateral custody attestation & proof-of-reserves. The rehypothecation guard (3.36) bounds a
counterparty's pledges against the capital it `held` — but that holdings figure was the one input the guard
**trusted**: it was *asserted*, not proven, so a counterparty over-stating its real reserves still passed the
guard, the way a self-asserted reputation score passed before attestation made standing verifiable. This release
makes the held capital itself **evidence-backed** — a signed, offline-verifiable **proof-of-reserves** the guard
reads as the held figure instead of the asserted default. Entirely additive and backward-compatible —
`API_VERSION` stays `3.0`, the existing escrow, pooling, rehypothecation, admission, and settlement paths are
unchanged (the `held=` input still works exactly as before), and the whole theme runs offline and deterministically.

### Added

- **CustodyAttestation (`vincio.settlement.custody`).** `attest_custody(poster, reserves, *, custodian=None,
  as_of=None)` issues a sealed, unsigned `CustodyAttestation` over the capital a poster actually holds — the
  proof-of-reserves analogue of `attest_reputation`. `reserves` is a number, a `{account: amount}` mapping, or
  `ReserveLine` / `(account, amount)` items (a negative holding is refused); the attested `reserves_usd` is their
  sum, re-derived on every verify. `custodian` defaults to the poster (self-custody when `custodian == poster`);
  otherwise an independent custodian vouches.
- **Read by the guard.** `guard_collateral(pools, *, poster=None, held=None, custody=None, verify_with=None)` reads
  a `custody` attestation's `reserves_usd` as the `held` figure (`reserves_proven` / `custodian` / `custody_hash` /
  `reserves_usd` on the `CollateralLedger`), bounding pledges against **proven** reserves. `held=` and `custody=`
  are mutually exclusive — the held figure has one source.
- **Under-reserved breach.** When the proven reserves fall below what the pools pledge, the shortfall surfaces as a
  bounded, pinpointed `UnderReservedBreach` (`.custodian` / `.attestation_hash` / `.reserves_usd` / `.pledged_usd` /
  `.shortfall_usd`) — exposed as `ledger.under_reserved` / `ledger.reserve_breach`, with `require_reserved()`
  raising on it. An asserted `held=` figure can over-commit but never *under-reserves*, because nothing proves it.
- **Offline-verifiable & refused-on-tamper.** `CustodyAttestation.verify(verifier=None, *, require=None)` →
  `CustodyAttestationVerification(valid, hash_ok, reserves_sound, signatures_ok, signed_by, reason)` recomputes the
  content hash and re-derives the reserve total from the line items, so a tampered figure is caught even after
  re-sealing; only the custodian signs (`sign(signer, *, party=None)`). The guard **refuses** a tampered reserve
  figure, a forged custodian (with `verify_with`), or an attestation that vouches for a different poster than the
  pools', and the under-reserved breach re-derives from the bytes (a fabricated or hidden breach is caught after
  re-sealing). `to_wire` / `from_wire` round-trip preserves verification.
- **App / book surface.** `app.attest_custody(...)` / `book.attest_custody(...)` sign the attestation as the
  custodian and record the issuance on the hash-chained audit log (action `custody_attestation`, decision =
  `self_custody` / `custodied`); `app.guard_collateral(..., custody=)` / `book.guard_collateral(..., custody=)` pass
  it through.
- **Public surface.** `CustodyAttestation`, `CustodyAttestationVerification`, `ReserveLine`, `UnderReservedBreach`,
  and `attest_custody` are exported from `vincio` and `vincio.settlement`.

### Benchmarks & SLOs

- **VincioBench `reputation_portability`** gains `proof_of_reserves_bounds_held`, `under_reserved_pinpoints`, and
  `por_auditable_offline`, gated by a new **proof-of-reserves** SLO (a custody attestation is read as the held
  figure, a proven shortfall is pinpointed as an under-reserved breach, and a tampered / forged / wrong-poster
  attestation is refused while the guard lands on the audit chain).

### Docs

- New runnable example `81_cross_org_custody_proof_of_reserves.py`; settlement guide, README, llms.txt, SECURITY,
  and ROADMAP updated; the next scheduled theme is **Cross-org custody liability attestation & proof-of-solvency**
  (target 3.38).

## [3.36.0] - 2026-06-23

Cross-org collateral rehypothecation guards & re-use bounds. A `CollateralPool` lets a counterparty back many
contracts with one posted stake — but a pool only ever **re-allocates** capital *within itself*. When the
counterparty pledges the **same** stake across more than one pool (or re-pledges collateral a beneficiary already
has a claim on), nothing bounded the **re-use**: the same capital was double-counted, over-stating what actually
backs each deal — the collateral analogue of a `SettlementRecord` double-counted before netting deduplicated it.
This release adds a **rehypothecation guard** — a bounded, offline-verifiable check that a posted stake is not
committed beyond what it holds across the pools that draw on it. Entirely additive and backward-compatible —
`API_VERSION` stays `3.0`, the existing escrow, pooling, admission, and settlement paths are unchanged, and the
whole theme runs offline and deterministically.

### Added

- **CollateralLedger (`vincio.settlement.rehypothecation`).** `guard_collateral(pools, *, poster=None, held=None,
  verify_with=None)` folds a counterparty's `CollateralPool`s into a sealed, unsigned `CollateralLedger` — the
  rehypothecation analogue of `net_settlements`. It reconciles what the pools collectively pledge (`pledged_usd`,
  the sum of their live balances) against the capital the poster actually holds (`held_usd`; defaults to the gross
  pledge minus the provably double-pledged capital, so a re-pledged contract surfaces by default while genuinely
  separately-funded pools do not). The poster defaults to the one every pool shares.
- **Cross-pool re-use bound.** A contract pledged across more than one pool is surfaced as a bounded, pinpointed
  `ReuseBreach` (`.contract_id` / `.pools` / `.pledged_usd` / `.secured_usd` / `.excess_usd`) — its collateral
  honorable only once, the excess provably double-pledged — and the aggregate over-commitment is `reuse_usd`
  (`.over_committed` / `.within_bounds` / `.status`), so the same capital pledged twice is named, never silently
  absorbed into an over-stated coverage figure.
- **Beneficiary-claim priority.** When a stake backs deals for more than one beneficiary, each `BeneficiaryClaim`
  (`.claim_usd` / `.secured_usd` / `.unsecured_usd` / `.share` / `.is_secured`) is bounded to its deterministic
  **pari-passu** share of the held capital (proportional to the capital pledged to it), so a forfeiture cannot pay
  one beneficiary out of capital another has first claim on.
- **Offline-verifiable.** The ledger reads only the existing signed, content-bound pools and asserts nothing it
  cannot recompute: a pool whose content hash no longer recomputes is **refused** at fold time (a forged pool
  signature too, with `verify_with`). It binds the poster, the folded per-pool figures, the reconciled totals, the
  breaches, and the claims onto a content hash (pools / breaches / claims sorted, so fold-order is irrelevant);
  `.verify(verifier=None, *, require=None)` → `CollateralLedgerVerification(valid, hash_ok, terms_sound,
  signatures_ok, signed_by, reason)` re-derives the re-use bound and the beneficiary apportionment, so a tampered
  total, breach, or claim is caught even after re-sealing. `.sign(signer, party)`, `.require_valid()`,
  `.require_within_bounds()` (raises if over-committed), `.to_wire` / `.from_wire`, and `.audit_details()` round it
  out.
- **Folds into the settlement path.** `app.guard_collateral(pools, *, poster=, held=, sign=True, verify_with=,
  record_audit=True)` and `book.guard_collateral(...)` sign the ledger as the org and record the guard on the
  hash-chained audit log (action `rehypothecation`, decision = `over_committed` / `within_bounds`).
- **Surface.** `from vincio import CollateralLedger, CollateralLedgerVerification, ReuseBreach, BeneficiaryClaim,
  guard_collateral` (all also in `vincio.settlement.__all__`, alongside the subpackage-only `LedgerPool` /
  `LedgerContract`); a `reputation_portability` VincioBench extension with four rehypothecation-guard metrics
  (reuse-bound-pinpoints, beneficiary-priority-bounded, guard-auditable-offline, guard-content-bound) and a
  `rehypothecation_guard` published SLO; and a runnable example,
  `examples/80_cross_org_collateral_rehypothecation.py`.

## [3.35.0] - 2026-06-23

Cross-org collateral pooling & cross-contract margin. An `Escrow` now backs *one* contract with collateral held
against its delivery — but a counterparty running many concurrent contracts had to lock **separate** collateral
per deal, even though its breaches and clean deliveries across those contracts net out. Capital was stranded
contract-by-contract the way bilateral settlements were stranded book-by-book before netting folded them. This
release adds a **bounded collateral pool** — a margin account a counterparty posts once that backs many contracts
at a deterministic, offline-verifiable allocation, the collateral analogue of the `NettingSet`. Entirely additive
and backward-compatible — `API_VERSION` stays `3.0`, the existing escrow, admission, and settlement paths are
unchanged, and the whole theme runs offline and deterministically.

### Added

- **CollateralPool (`vincio.settlement.collateral`).** `post_collateral_pool(contracts, *, poster=None,
  posted=None, decisions=None, fraction=None, config=None)` binds a counterparty's single posted stake to the set
  of contracts it backs into a sealed, unsigned `CollateralPool` — the collateral analogue of a `NettingSet`. Each
  contract's required collateral re-derives via the same source resolver as `post_escrow` (a matching
  `AdmissionDecision` in `decisions` — a dict keyed by contract id or one decision for all — a uniform `fraction`,
  or the admission posture stamped onto each contract's terms), and each is allocated a per-contract share
  **proportional to its required collateral**. The poster defaults to the common seller; the posted stake defaults
  to the total required collateral, so the pool starts exactly collateralized.
- **Deterministic draw, release & top-up.** `CollateralPool.draw(record, *, config=None)` / `draw_pool(pool,
  record, *, config=None)` settles one backed contract against its `SettlementRecord`: a clean delivery
  **releases** its requirement back to the available balance (freeing capital for the next contract), and a breach
  **draws** a bounded, pinpointed slice — `min(shortfall, max_forfeit_fraction)` of its required collateral,
  pinpointed in `.breaches` — from the shared stake, releasing the rest (never the whole stake, never punitive),
  driven by the *same* `SettlementRecord` verdict the books close on. `CollateralPool.back(contract, ...)` adds a
  contract to the open pool, and a pool committed below the collateral its open contracts require surfaces a
  bounded `.topup_usd` obligation that `CollateralPool.top_up(amount)` clears. The forfeiture policy reuses
  `EscrowConfig(max_forfeit_fraction=1.0)`.
- **Offline-verifiable.** `CollateralPool` binds the poster, the posted stake, every backed contract, and the
  balances onto a content hash (contracts sorted by id, so add-order is irrelevant); `.verify(verifier=None, *,
  require=None)` → `CollateralPoolVerification(valid, hash_ok, terms_sound, signatures_ok, signed_by, reason)`
  re-derives every allocation and reconciles the balance (`balance == posted − drawn`, the top-up and each
  forfeiture re-derive), so a tampered allocation, balance, or forfeiture is caught even after re-sealing.
  `.sign(signer, party)` (the poster or a counterparty only), `.require_valid()`, `.to_wire` / `.from_wire`, and
  `.audit_details()` round it out; `.draw` is idempotent-guarded and contract-matched.
- **Folds into the settlement path.** `app.post_collateral_pool(...)` posts and audits the pool; `app.settle(
  contract, ..., pool=None)` draws an attached pool against the record it produces in the same call; `app.draw_pool(
  pool, record, ...)` draws one against a record you already have — every post, draw, release, and top-up signed
  and recorded on the hash-chained audit log (action `collateral_pool`, decision = the status).
- **Surface.** `from vincio import CollateralPool, CollateralPoolVerification, PooledContract,
  post_collateral_pool, draw_pool` (all also in `vincio.settlement.__all__`, alongside `PooledContractState` /
  `PoolStatus`); a `reputation_portability` VincioBench extension with five collateral-pool metrics
  (allocates-proportionally, draws-and-frees, top-up-surfaces, auditable-offline, content-bound) and a
  `collateral_pooling` published SLO; and a runnable example,
  `examples/79_cross_org_collateral_pooling.py`.

## [3.34.0] - 2026-06-22

Cross-org collateralized settlement & escrow. Admission now sets a required collateral / escrow fraction on a
thin or low-trust counterparty's contract — but the fraction was still only a *number stamped on the terms*;
nothing **held** it, released it on a clean delivery, or forfeited a slice on a breach. A counterparty admitted
on conservative terms posted no actual collateral, so the escrow the admission policy asked for had no teeth,
and a breach was debited only to reputation after the fact. This release makes the posted collateral a
**verifiable, offline escrow bound to the contract** — held against delivery and settled deterministically — so
the conservative terms a thin standing is admitted on are backed by something, not merely recorded. Entirely
additive and backward-compatible — `API_VERSION` stays `3.0`, the existing negotiation, contracting, and
settlement paths are unchanged, and the whole theme runs offline and deterministically.

### Added

- **Escrow (`vincio.settlement.escrow`).** `post_escrow(contract, *, decision=None, fraction=None, amount=None,
  poster=None, beneficiary=None, config=None)` binds an admission-required collateral amount to a *specific*
  `Contract` and counterparty into a sealed, unsigned `Escrow` — the escrow analogue of a `SettlementRecord`.
  The held amount comes from an explicit `amount` (a flat stake), an explicit `fraction` of the contract price,
  an `AdmissionDecision.escrow_fraction` (`decision=`), or the admission posture `apply_to_terms` already
  stamped onto the contract's terms; the poster defaults to the seller (the counterparty backing its delivery)
  and the beneficiary to the buyer.
- **Deterministic release & forfeiture.** `Escrow.resolve(record, *, config=None)` / `settle_escrow(escrow,
  record, *, config=None)` settles the escrow against the contract's `SettlementRecord`: a fulfilled delivery
  **releases** the whole stake back to the poster, and a breach **forfeits** `min(shortfall,
  max_forfeit_fraction)` of the stake — the per-dimension shortfall being how far delivery missed the worst
  breached term, pinpointed in `.breaches` — releasing the remainder (never the whole stake, never punitive).
  The outcome is driven by the *same* `SettlementRecord` verdict the books already close on. `EscrowConfig(
  max_forfeit_fraction=1.0)` caps a single breach's forfeiture (set `<1` for a guaranteed residual).
- **Offline-verifiable.** `Escrow` binds the contract, the amount, the admission posture, and the disposition
  onto a content hash; `.verify(verifier=None, *, require=None)` → `EscrowVerification(valid, hash_ok,
  terms_sound, signatures_ok, signed_by, reason)` re-derives the held amount from the fraction and the
  release / forfeit split from the shortfall, so a tampered amount or forfeiture is caught even after
  re-sealing. `.sign(signer, party)` (buyer/seller only), `.require_valid()`, `.to_wire` / `.from_wire`, and
  `.audit_details()` round it out; resolution is idempotent-guarded and contract-matched.
- **Folds into the settlement path.** `app.post_escrow(...)` posts and audits the collateral; `app.settle(
  contract, ..., escrow=None, escrow_config=None)` resolves an attached escrow against the record it produces in
  the same call; `app.settle_escrow(escrow, record, ...)` resolves one against a record you already have —
  every post, release, and forfeiture signed and recorded on the hash-chained audit log (action `escrow`,
  decision = the state).
- **Surface.** `from vincio import Escrow, EscrowConfig, EscrowVerification, post_escrow, settle_escrow` (all
  also in `vincio.settlement.__all__`, alongside `EscrowState` / `EscrowSignature`); a `reputation_portability`
  VincioBench extension with five escrow metrics (posts-against-contract, releases-on-fulfilment,
  forfeits-proportional-to-breach, auditable-offline, folds-into-settlement-path) and an `escrow_settlement`
  published SLO; and a runnable example, `examples/78_cross_org_collateralized_escrow.py`.

## [3.33.0] - 2026-06-22

Cross-org reputation-gated admission & progressive exposure. Reputation is now portable, current,
discoverable, and trust-weighted — but it was still only ever *consulted* as a soft weight on a negotiation;
nothing **acted** on a too-thin or too-low standing to bound how much a new counterparty was trusted with up
front. A brand-new or low-trust counterparty was admitted to a contract on the same terms as a long-trusted
one, the regression caught only after the fact. This release turns the weighted standing into a **graduated
admission posture** — bounding a counterparty's exposure to what its earned trust justifies, ramping it as
trust accrues — so onboarding an unknown org is safe by construction. Entirely additive and
backward-compatible — `API_VERSION` stays `3.0`, the existing negotiation, contracting, and settlement paths
are unchanged, and the whole theme runs offline and deterministically.

### Added

- **Admission policy (`vincio.settlement.admission`).** `AdmissionConfig(parity_exposure_usd=1000.0,
  floor_fraction=0.1, full_trust_evidence=10.0, ramp_floor=0.2, max_escrow_fraction=0.5, min_sla_factor=0.5)`
  configures a graduated-exposure map; `AdmissionPolicy(config=None).admit(subject, *, reputation=None,
  ledger=None, standing=None)` (and the module-level `admit(...)`) reads a counterparty's standing — from an
  imported `PortableReputation` or a local `ReputationLedger` — and maps it to a bounded `AdmissionDecision`.
- **Reputation-gated terms.** Exposure is the product of two bounded signals — the standing's posterior-mean
  reputation and a ramp over its corroborated, settled evidence — lifted off `floor_fraction`, so a thin or
  low-trust standing is admitted on *conservative* terms rather than refused (discounted exposure, never a
  hard gate, never singled out). The decision carries a `max_contract_value_usd` exposure ceiling, an
  `escrow_fraction` (collateral demanded, falling to `0` at parity), and an `sla_factor` (SLA tightening,
  relaxing to `1` at parity).
- **Progressive ramp.** `AdmissionConfig.ramp_progress(evidence)` climbs `[ramp_floor, 1]` to parity at
  `full_trust_evidence` settled deliveries (and never past it), so a counterparty's ceiling **ramps**
  deterministically toward parity as it accrues history and a regression walks it back — bounded and
  reversible. Local first-hand evidence wins over what others attest: when a portable prior's `base` ledger
  has earned evidence for the subject, the standing is read from that ledger, exactly as
  `PortableReputation.weight` resolves it, so a regression the importer lived through walks exposure back.
- **Offline-verifiable.** `AdmissionDecision` binds the `Standing` it read and the terms it set onto a content
  hash; `.verify()` → `AdmissionVerification(valid, hash_ok, terms_sound, reason)` re-derives the terms from
  the bound standing, so a tampered ceiling, escrow, or SLA factor is caught even after re-sealing.
  `.require_valid()`, `.to_wire` / `.from_wire`, and `.audit_details()` round it out.
- **Folds into the existing path.** `AdmissionDecision.bound_position(position)` clamps a buyer's
  `NegotiationPosition` price reservation to the exposure ceiling and tightens its SLA reservation (a copy;
  the original is untouched), so the bargain can only converge within the admitted exposure;
  `.apply_to_terms(terms)` caps a `ContractTerms` price / SLA and stamps the escrow posture into the terms'
  `metadata` (excluded from the contract's canonical hash, so a contract minted from the capped terms stays
  offline-verifiable).
- **App surface.** `app.admit(subject, *, reputation=None, policy=None, config=None, record_audit=True)` reads
  the same source the negotiation path weights by (`imported_reputation` else `reputation_ledger`) and records
  the decision on the hash-chained audit log (action `reputation_admission`, decision `parity` | `graduated`).
- **Surface.** `from vincio import AdmissionConfig, AdmissionDecision, AdmissionPolicy, AdmissionVerification,
  admit` (all also in `vincio.settlement.__all__`, alongside `Standing`); a `reputation_portability`
  VincioBench extension with five admission SLOs (gates-by-reputation, ramps-progressively,
  newcomer-conservative, auditable-offline, folds-into-path) and a `reputation_gated_admission` published SLO;
  and a runnable example, `examples/77_cross_org_reputation_gated_admission.py`.

## [3.32.0] - 2026-06-22

Cross-org transitive trust & Sybil-resistant attestation weighting. Reputation is now portable, current,
discoverable, and revocable — but every counted issuer's evidence pooled into the prior with **equal
pull**, weighted only by *how much* it attests, not by *how much the importer trusts the issuer*. A clutch
of unknown peers could therefore out-evidence a few an importer has lived through, and an adversary could
spin up **Sybil** issuers that all vouch the same way. This release adds an opt-in, bounded, transitive
web-of-trust that scales each issuer's contributed evidence by the importer's **own trust in that issuer**,
so pull follows earned trust rather than issuer count, without a central trust authority. Entirely additive
and backward-compatible — `API_VERSION` stays `3.0`, and with no trust source the combination pools with
equal pull byte-for-byte as before; the whole theme runs offline and deterministically.

### Added

- **Trust kernel (`vincio.settlement.attestation`).** `TrustConfig(max_depth=1, hop_decay=0.5,
  trust_floor=0.1, trust_ceiling=1.0)` configures a bounded, transitive web-of-trust;
  `build_trust_model(attestations, *, base=None, config=None, attestation_config=None, verify_with=None)`
  builds a `TrustModel` from the importer's own `ReputationLedger` and the attestations on hand. **Hop 0:**
  an issuer the importer has first-hand evidence for is trusted as much as that ledger weights it. **Hops
  1..max_depth:** an already-trusted issuer that *attests another issuer* (vouches for it as a counterparty)
  lends it trust derived from that pooled standing, attenuated by `hop_decay` per hop, under a hard depth
  bound. **Unreached:** an issuer neither known nor reachable from a trusted root falls back to the floor —
  counted, never zeroed. Only admissible (verified) attestations vouch, and an issuer never bootstraps its
  own trust.
- **Sybil resistance.** Trust is lent only *outward from a trusted root*, so a cluster of mutually-vouching
  unknown issuers is never reached and every member stays at the floor — corroboration from a few trusted
  peers cannot be outvoted by volume from unknown ones.
- **Issuer-weighted pooling.** `combine_attestations(attestations, *, ..., trust=None, trust_config=None)`
  scales each issuer's contributed evidence *mass* (successes and failures together, so it changes how much
  an issuer *pulls*, never the reputation it attests) by the resolved trust multiplier, bounded
  `[trust_floor, 1]`. Pass a `trust` source (a `TrustModel`, anything exposing `trust_in` / `weight`, or an
  `issuer -> float` callable) or a `trust_config` to build the model automatically from `base` and the full
  attestation set. The applied multiplier is pinpointed on `AttestationVerdict.trust` (counted) and
  `SubjectStanding.issuer_trust`; `PortableReputation.trust` holds the model and `.trust_in(issuer)` reads it.
- **App surface.** `app.import_reputation(..., trust=None, trust_config=None)` and
  `app.gather_reputation(...)` / `app.agather_reputation(...)` thread the trust source through, rooted in
  `self.reputation_ledger`. `TrustModel` quacks like a ledger (`weight(issuer)` aliases `trust_in(issuer)`),
  and each issuer's `IssuerTrust` records its `trust` / `depth` / `vouched_by` so a multiplier is always
  traceable.
- **Surface.** `from vincio import TrustConfig, TrustModel, IssuerTrust, build_trust_model` (all also in
  `vincio.settlement.__all__`); a `reputation_portability` VincioBench extension with a
  `reputation_transitive_trust` SLO; and a runnable example,
  `examples/76_cross_org_transitive_trust.py`.

## [3.31.0] - 2026-06-22

Cross-org reputation gossip & attestation exchange. Attestations are now portable, time-aware, and
revocable — but an importer still had to be *handed* the right bundle out of band: it had no way to
**discover** who has attested a counterparty, or to learn that an issuer has since revoked one, without a
hosted registry. This release adds the discovery analogue for reputation: a bounded, **pull-based**
exchange of the existing signed artifacts over the A2A fabric, so an importer assembles a *current* prior
from what its peers hold, never from a central bulletin board. Entirely additive and
backward-compatible — `API_VERSION` stays `3.0`, the existing attestation, revocation, and negotiation
paths are unchanged, and the whole theme runs offline and deterministically.

### Added

- **Reputation exchange (`vincio.settlement.exchange`).** A `ReputationBundle` is the signed artifacts a
  peer holds about a subject; `attestation_a2a_server(book, *, revocations=None, attestations=None,
  config=None, ...)` exposes an org's settlement book as a queryable A2A peer whose Agent Card advertises
  an `attestation-exchange` skill. **Pull, never push:** answering a subject query, the peer returns only
  its own signed artifacts — the current attestation it can issue from its `SettlementBook` records, plus
  the revocations it has signed (or an explicit signed snapshot). A subject it has no admissible history
  for yields an attestation-free bundle rather than an error.
- **Bounded, governed gather.** `AttestationExchange(client, *, peer_id="")` pulls one peer
  (`.fetch(subject)`); `gather_reputation(subject, *, peers, directory=None, config=None, verify_with=None,
  base=None, allow_self=False, held_attestations=None, held_revocations=None, as_of=None, max_peers=None,
  audit=None, record_audit=True)` visits a **bounded** set of peers in deterministic order, **governs**
  each through an `AgentDirectory` allow-list (a denied peer skipped and pinpointed, its resolution
  audited), **verifies** every fetched artifact from the bytes (a forged or tampered one refused),
  **deduplicates** by content hash, and folds the gathered (plus any already-held) artifacts into the
  *same* `combine_attestations` under the same freshness, revocation, and `[floor, 1]` discipline. Returns
  a `GatheredReputation` exposing `weight(member_id)` / `standing(id)` (delegating to the assembled
  `PortableReputation`), the per-peer `PeerVisit` record, and the deduplicated artifacts.
- **Auditable & offline.** Every peer visited (`reputation_peer`) and every artifact fetched
  (`reputation_fetch`) lands on the hash-chained audit log; the whole exchange runs byte-for-byte the same
  against deterministic in-process peers (`connect_a2a_in_process`) as over the live fabric.
- **App surface.** `app.serve_attestations(*, book=None, revocations=None, attestations=None, config=None,
  ...)` exposes this app's book as a peer (returning its retained revocations); `app.gather_reputation(...)`
  / `app.agather_reputation(...)` gather with `base=self.reputation_ledger` and (with `weight=True`, the
  default) attach the assembled prior so the next `app.negotiate` weights an unknown counterparty by what
  its peers attest. `app.revoke_attestation` now retains the issued revocation so `serve_attestations` can
  gossip it.
- **Surface.** `from vincio import AttestationExchange, ReputationBundle, PeerVisit, GatheredReputation,
  attestation_a2a_server, gather_reputation` (all also in `vincio.settlement.__all__`); a
  `reputation_portability` VincioBench extension with a `reputation_exchange` SLO; and a runnable example,
  `examples/75_cross_org_reputation_gossip.py`.

## [3.30.0] - 2026-06-22

Cross-org attestation revocation & freshness. A `ReputationAttestation` is a point-in-time claim, but
standing changes — a counterparty reliable a year ago may have regressed, and an issuer may need to
**withdraw** a claim it can no longer stand behind — and the portable prior would otherwise trust a
signed attestation forever. This release makes portable reputation **time-aware and revocable**, so an
imported prior reflects *current* standing, not a frozen snapshot, without becoming a hosted revocation
service. Entirely additive and backward-compatible — `API_VERSION` stays `3.0`, an attestation with no
declared validity window hashes and verifies exactly as before, a combination with no as-of clock is
point-in-time, and the whole theme runs offline and deterministically.

### Added

- **Freshness — a validity window on the attestation.** `attest_reputation(..., horizon_days=None)`
  (also `app.attest_reputation` / `book.attest`) lets an issuer declare how long its attestation holds;
  the window is bound into the signed attestation hash only when set, so a no-horizon attestation hashes
  exactly as it did before. `attestation.expires_at`, `.is_stale(as_of)`, and `.age_days(as_of)` expose
  it. Against an as-of clock, `combine_attestations(..., as_of=)` **excludes** a stale attestation and
  pinpoints it (`PortableReputation.stale`, `AttestationVerdict.stale`).
- **Freshness — half-life decay.** `AttestationConfig(half_life_days=None)` decays an older (but still
  valid) attestation's evidence by age — `0.5 ** (age_days / half_life_days)` of its mass, its attested
  ratio preserved — so an old attestation decays out of the pooled prior toward the benefit-of-the-doubt
  rather than anchoring it forever. With no as-of clock, attested evidence is never decayed.
- **Revocation — a content-bound, offline-verifiable `AttestationRevocation`.**
  `revoke_attestation(attestation_or_hash, *, subject="", issuer="", replacement=None, reason="")` (also
  `app.revoke_attestation` / `book.revoke`) issues a signed revocation that withdraws or supersedes a
  prior attestation **by its hash**. The revocation hash binds the issuer, the subject, the withdrawn
  hash, and any replacement; `revocation.sign(signer, party=None)` co-signs it and
  `revocation.verify(verifier=, require=None)` → `RevocationVerification` recomputes it offline.
  `.revokes(attestation)`, `.is_supersession`, `.require_valid()`, `.to_wire` / `.from_wire`,
  `.print_summary()`.
- **Revocation folded into the combination.** `combine_attestations(..., revocations=)` (also
  `app.import_reputation(..., revocations=, as_of=)`) excludes an attestation an admissible,
  **issuer-matched** revocation withdraws — pinpointed (`PortableReputation.revoked`,
  `AttestationVerdict.revoked`), never silently honored. A revocation is honored only when it verifies
  (and, with a verifier, the issuer signature checks) and is issued by the same party whose attestation
  it names, so a forged revocation, or one naming another org's attestation, **cannot cancel a claim**.
  `app.revoke_attestation` records the issuance on the audit chain (action `attestation_revocation`).
- **Surface.** `from vincio import AttestationRevocation, revoke_attestation` (and
  `vincio.settlement.RevocationVerification`); a `reputation_portability` VincioBench extension with a
  freshness + revocation SLO (`attestation_freshness_and_revocation`); and a runnable example,
  `examples/74_cross_org_attestation_revocation_freshness.py`.

## [3.29.0] - 2026-06-22

Cross-org reputation attestation & portability. Settlement, netting, and arbitration all close the
reputation loop, but the standing they earn lives inside one org's own `ReputationLedger` — a *new*
counterparty, with no prior history, has no way to trust it without a hosted reputation bureau. This
release adds the last rung: making earned standing **portable**. An org issues a signed,
offline-verifiable attestation over a counterparty's standing, a prospective counterparty verifies it
from the bytes alone, and several issuers' attestations combine into a bounded, evidence-weighted prior
that weights the next negotiation — reputation that travels the fabric, never a central service.
Entirely additive and backward-compatible — `API_VERSION` stays `3.0`, the existing settlement and
negotiation paths are unchanged, and the whole theme runs offline and deterministically.

### Added

- **Reputation attestation (`vincio.settlement.attestation`).** `attest_reputation(records, subject,
  *, issuer="", resolutions=None, config=None, verify_with=None, note="")` issues a
  `ReputationAttestation` over a counterparty's earned standing, derived only from an org's own signed
  `SettlementRecord`s (counting the ones where the subject was the **seller** — a fulfilled settlement
  a success, a breach a failure) and arbitration `Resolution`s (a dissent a failure). It reads only
  what it can recompute: a record whose reconciliation hash no longer recomputes (or, with a verifier,
  whose signature is forged) is skipped, and the exact source hashes are bound. Raises `SettlementError`
  when there is no admissible history to attest.
- **A content-bound, offline-verifiable `ReputationAttestation`.** An attestation hash binds the issuer,
  the subject, the evidence counts, the prior, and the source hashes (the id and timestamp excluded; the
  issuer **is** bound — an attestation is one issuer's signed claim); `attestation.sign(signer,
  party=None)` co-signs it and `attestation.verify(verifier=, require=None)` → `AttestationVerification`
  recomputes it offline and **re-derives the attested reputation from the evidence counts**
  (`evidence_sound`), so a tampered score is caught even after re-sealing and a forged issuer is caught.
  `.require_valid()`, `.to_wire` / `.from_wire`, `.print_summary()`.
- **Combining into an evidence-weighted prior.** `combine_attestations(attestations, *, subject=None,
  config=None, verify_with=None, base=None, allow_self=False)` pools several issuers' attestations into a
  bounded `PortableReputation` prior. Because a Beta-Bernoulli posterior is conjugate, combining is
  *pooling the evidence*, never a single self-asserted number: an issuer that vouches for itself is
  **refused**, an issuer cannot stack its own pull (only its largest attestation for a subject is
  counted), a tampered or forged attestation is **pinpointed** (`AttestationVerdict`) and excluded, an
  optional `AttestationConfig.per_issuer_cap` bounds any one issuer's mass, and the importer's own prior
  anchors the pooled posterior. The prior exposes `weight(member_id)` ∈ `[floor, 1]`, so it drops into
  the existing negotiation/discovery path unchanged; with a local `ReputationLedger` as the `base`, a
  counterparty the importer already knows keeps its own earned standing and only an unknown one leans on
  the imported attestations.
- **Attestation on the app & book surface.** `app.attest_reputation(subject, *, book=None,
  resolutions=None, config=None, sign=True, record_audit=True)` issues from this app's settlement book,
  signs as the app, and records the issuance on the audit chain (action `reputation_attestation`);
  `app.import_reputation(attestations, *, subject=None, config=None, verify_with=None, allow_self=False,
  weight=True)` combines a bundle and (by default) attaches the prior so the next `app.negotiate` weights
  a counterparty with no local history by what its past counterparties attest. `book.attest(subject, ...)`
  issues signed as the book owner.
- **A `reputation_portability` VincioBench family** holding an attestation-correctness SLO (an
  attestation summarizes the issuer's own earned outcomes, several issuers' evidence pools into one
  bounded prior, a self-attestation and a stacked one are refused, an unknown counterparty falls back to
  the prior, and the prior weights a negotiation) and an attestation-integrity SLO (an attestation signs
  and verifies offline, a tampered score is caught even after re-sealing because the reputation
  re-derives from the evidence, a forged issuer is refused, two importers compute the same standing, and
  issuance is audited).
- **Example `73_cross_org_reputation_attestation.py`** and a reputation-portability section in the
  settlement guide.

### Public surface

- Added to `vincio.__all__`: `ReputationAttestation`, `PortableReputation`, `attest_reputation`,
  `combine_attestations`. The supporting types (`AttestationConfig`, `AttestationVerification`,
  `AttestationVerdict`, `SubjectStanding`) are exported from `vincio.settlement`. `API_VERSION` remains
  `3.0`.

## [3.28.0] - 2026-06-22

Cross-org dispute resolution & arbitration. With settlements signed, reconciled, netted, and a
disagreement *pinpointed* as a `NettingDispute`, this release adds the next rung: **resolving** it.
Each party submits its signed `SettlementRecord`s for the disputed contract and a deterministic
adjudication decides which figure stands — a library-side protocol, never a hosted arbitration
service or a court of record. The decision rests on nothing it cannot recompute: a reconciliation
hash both parties co-signed is upheld, a contradicting unilateral claim is rejected and pinpointed, a
tampered claim is marked inadmissible, and a genuine standoff is honestly left unresolved. The
resulting `Resolution` is content-bound and verifies offline the way a settlement record does.
Entirely additive and backward-compatible — `API_VERSION` stays `3.0`, the existing settlement path
is unchanged, and the whole theme runs offline and deterministically.

### Added

- **Dispute arbitration (`vincio.settlement.arbitration`).** `arbitrate(records, *, contract_id=None,
  arbiter="", verify_with=None)` adjudicates a disputed contract from the signed records its parties
  submit and returns a `Resolution`. The decision is deterministic and evidence-based: a
  reconciliation hash that **both** the buyer and the seller signed (each on their own record,
  co-signing one figure) is mutually corroborated and **upheld**; a unilateral claim contradicting it
  is **rejected** and its claimant pinpointed; a single uncontested figure stands on its own; and when
  neither side's figure is corroborated the dispute is left **unresolved** rather than decided by fiat
  (`status` is `"upheld"` | `"unresolved"`).
- **Inadmissible claims are pinpointed, never raised.** Unlike netting, which *refuses* to clear over
  a tampered book, arbitration is the venue where a bad claim is adjudicated: a claim whose
  reconciliation hash no longer recomputes, one carrying no signature, or — with a verifier — one with
  a forged signature is marked **inadmissible** with a reason on its `ClaimVerdict`, never silently
  dropped and never crashing the resolution.
- **A content-bound, offline-verifiable `Resolution`.** A resolution hash binds the contract, the
  parties, the outcome, and every adjudicated claim (by reconciliation hash, corroborating signers,
  admissibility, and whether it stands — not by record id, so the same claim from both sides binds
  once); `resolution.sign(signer, party=)` co-signs it and `resolution.verify(verifier=, require=)` →
  `ResolutionVerification` recomputes it offline and **re-derives the whole decision from the recorded
  claims** (`decision_sound`), so a flipped verdict is caught even after re-sealing. Because the hash
  excludes the arbiter, two arbiters reading the same records compute the same co-signable hash.
  `.require_valid()` / `.require_resolved()`, `.standing_claims` / `.rejected_claims` /
  `.inadmissible_claims` / `.dissenters`.
- **Arbitration on the app & book surface.** `app.arbitrate(records, *, contract_id=None, sign=True,
  verify_with=None, record_audit=True, record_reputation=True)` adjudicates, signs the resolution as
  the app, records it on the audit chain (action `arbitration`), and **closes the reputation loop** by
  debiting each dissenter (the party whose admissible claim did not stand; an unresolved standoff
  debits nobody). `book.arbitrate(*counterparty_records, contract_id=None, sign=True, verify_with=None)`
  resolves one org's own record against a counterparty's submitted claims.
- **An `arbitration` VincioBench family** holding a resolution-correctness SLO (a co-signed figure is
  upheld, a contradicting claim is rejected and its claimant pinpointed, a standoff is left unresolved,
  and a tampered claim is inadmissible) and a resolution-integrity SLO (the resolution signs and
  verifies offline, a tampered verdict is caught even after re-sealing because the decision re-derives
  from the claims, two arbiters compute the same co-signable hash, and the adjudication is audited and
  debits the dissenter).
- **Example `72_cross_org_dispute_arbitration.py`** and a dispute-resolution section in the settlement
  guide.

### Public surface

- Added to `vincio.__all__`: `Resolution`, `arbitrate`. The supporting types (`ResolutionStatus`,
  `ResolutionVerification`, `ClaimVerdict`) are exported from `vincio.settlement`. `API_VERSION`
  remains `3.0`.

## [3.27.0] - 2026-06-22

Cross-org settlement netting & multilateral clearing. With bilateral settlements signed, reconciled,
and reputation-closing, this release adds the next rung: **netting** them. An org is often both a
buyer and a seller across a web of contracts; netting folds a fleet's many bilateral `SettlementBook`
balances into a single minimal set of net obligations, so the books close once — a library-side
clearing *calculation*, never a hosted clearing house or a payment rail. The cleared `NettingSet` is
content-bound and verifies offline the way a settlement record does. Entirely additive and
backward-compatible — `API_VERSION` stays `3.0`, the existing settlement path is unchanged, and the
whole theme runs offline and deterministically.

### Added

- **Multilateral netting (`vincio.settlement.netting`).** `net_settlements(records, *, owner=,
  fleet=, verify_with=)` and `net_books(books, *, owner=, verify_with=, require_intact=)` fold a
  fleet's signed settlement records into a `NettingSet`. Each settled contract is a directed payable
  (the buyer owes the seller the agreed price for the scope); the same settlement seen from both
  books is **deduplicated by its reconciliation hash, not double-counted**, the directed payables
  aggregate into `GrossObligation`s per pair, collapse to one `BilateralNet` figure per counterparty,
  and the per-org `NetPosition`s (which sum to zero) **clear** to the minimal set of `NetObligation`
  transfers — at most `N − 1` for `N` parties, net-debtors paying net-creditors, deterministically
  (ties broken by org id).
- **A content-bound, offline-verifiable `NettingSet`.** A netting hash binds the fleet, the exact
  source records read, the net positions, and the cleared obligations; `netting.sign(signer, party=)`
  co-signs it and `netting.verify(verifier=, require=)` → `NettingVerification` recomputes it offline
  — the hash matches, the positions balance to zero (`positions_balanced`), and the cleared transfers
  reproduce every position (`conserves`). A tampered source record is **refused** (`SettlementError`),
  and two books that disagree on a contract are pinpointed as a `NettingDispute` and excluded
  (`.clean` / `.require_clean()`), never silently absorbed.
- **Netting on the app & book surface.** `app.clear_settlements(*, books=, records=, sign=True,
  verify_with=, record_audit=True)` nets a fleet (defaulting to the app's attached book), signs the
  set as the app, and records it on the audit chain (action `netting`); `book.net(*, sign=True)` nets
  one org's own book into its position against each counterparty.
- **A `netting` VincioBench family** holding a netting-correctness SLO (the net positions balance to
  zero, the cleared obligations reproduce them, and a cycle clears to fewer transfers than its gross
  edges) and a netting-integrity SLO (the cleared set signs and verifies offline, a tampered figure
  or source record is caught, two clearers compute the same co-signable hash, and a disagreement is
  pinpointed as a dispute).
- **Example `71_cross_org_settlement_netting.py`** and a netting section in the settlement guide.

### Public surface

- Added to `vincio.__all__`: `NettingSet`, `net_settlements`, `net_books`. The supporting models
  (`NetPosition`, `NetObligation`, `BilateralNet`, `GrossObligation`, `NettingDispute`,
  `NettingVerification`) are exported from `vincio.settlement`. `API_VERSION` remains `3.0`.

## [3.26.0] - 2026-06-22

Cross-org workflow discovery & dynamic choreography. With cross-org sagas negotiated, contracted,
settled, and reconciled, this release adds the next rung: **who** runs each step, resolved at run
time rather than wired by org id up front. A saga step declares the *capability* it needs and the
engine resolves the counterparty at dispatch time from the governed `AgentDirectory` — ranked by
reputation and prior settlement fit — so a choreography binds the best-available counterparty for
each step, never a hosted matching service. Discovery changes *who* runs a step, never *how*: the
resolved org runs under the same allow-list, contract, per-org audit, compensation, durability, and
A2A portability a statically-wired one does. Entirely additive and backward-compatible —
`API_VERSION` stays `3.0`, the default static-wiring path is unchanged, and the whole theme runs
offline against deterministic local participants.

### Added

- **Run-time capability binding (`vincio.choreography.discovery`).** A `Saga` step may declare the
  `capability=` it needs instead of a fixed `participant=`; a `CapabilityBinder(directory, *,
  reputation=, settlement_book=, weights=)` resolves it to a participant at dispatch time. `.bind(
  step, *, available=)` finds the directory records advertising the capability, governs each through
  the directory's allow-list (audited), keeps the allowed **and** reachable candidates, and ranks
  them by a weighted mean (`BindingWeights`) of reputation weight, prior settlement reliability, and
  contract fit — best first, ties broken deterministically by org id — returning a `StepBinding`
  (chosen org + the full ranked `BindingCandidate` field, for audit).
- **Discovery on the app surface.** `app.choreograph(saga, *, participants=, directory=, binder=,
  binding_weights=)` / `aresume_choreography(...)` build the binder automatically from `directory=`
  and the app's reputation ledger and settlement book (or accept a prepared `binder=`). The binding
  decision is recorded on the saga journal (`result.bindings` / `journal.bindings()`) and on the
  coordinator's hash-chained audit chain (`choreography_bind`).
- **A `discovery` VincioBench family** holding a binding-correctness SLO (the best-ranked allowed
  candidate is bound, deterministically, and recorded) and a governance-preservation SLO (an unlisted
  or unreachable candidate is never bound, every resolution and the binding are audited, a capability
  no eligible candidate advertises is refused, and the bound step is contract-enforced, compensated
  at the bound org, durable, and A2A-portable as a static one).
- **Example `70_cross_org_workflow_discovery.py`** and a discovery section in the choreography guide.

### Changed

- `SagaStep` accepts exactly one of `participant=` (static) or `capability=` (discovered);
  `Saga.step(...)` gains `capability=`. `StepRecord` / `StepRequest` carry the bound `capability`,
  and `StepRecord.binding` carries the `StepBinding`. A discovered step is **compensated at the org it
  was bound to** (recorded on the journal, never re-resolved), and a resume re-binds only steps not
  yet run. Fully backward-compatible: a statically-wired saga behaves exactly as before.

### Public surface

- Added to `vincio.__all__`: `CapabilityBinder`, `BindingWeights`, `BindingCandidate`, `StepBinding`.
  `API_VERSION` remains `3.0`.

## [3.25.0] - 2026-06-22

Agent-to-agent settlement & metering. With cross-org sagas dispatching contracted work across
organizations, this release adds the next rung: **closing the books** on it — a metered, auditable
settlement record reconciling delivered work against a negotiated `Contract`, the way a run closes
its cost report. It is never a payment rail, only a verifiable ledger of what was owed and
delivered: usage accrues against the agreed price as the work completes, a typed, signed settlement
record reconciles delivery against the terms and verifies offline from the bytes alone, two orgs'
records reconcile across the boundary, and a settled overrun or shortfall closes the reputation
loop. Landed in the *same* governed, audited runtime, never as a hosted marketplace or a payment
processor. Entirely additive and backward-compatible — `API_VERSION` stays `3.0`, and the whole
theme runs offline against deterministic figures.

### Added

- **The settlement package (`vincio.settlement`).** A `Meter` (`app.meter(contract)`) accrues the
  usage of work delivered under a negotiated `Contract` — each unit a `UsageEvent` attributed to the
  contract and the run — into a deterministic, **total-preserving** `MeterReading` (cost and latency
  summed, quality the minimum/weakest link, totals exactly the sum of the events); `Meter.from_saga`
  builds a meter per contract from a saga's durable journal.
- **Signed, offline-verifiable settlement.** `app.settle(contract, *, reading= | cost_usd=,
  latency_ms=, quality=, party=, sign=, record_reputation=)` reconciles delivery against the agreed
  price / SLA / quality (via `contract.check`) into a `SettlementRecord` — `.status`
  settled|breached, `.amount_owed_usd`, `.balance_usd` (+credit / −overrun), per-dimension `.lines`,
  `.breaches`. Both parties sign one *reconciliation hash* over the economic facts (run-id- and
  timestamp-independent, so two sides co-sign the same hash); `record.verify(verifier, require=)`
  recomputes it from the bytes alone, so a tampered figure or forged signature is caught. A breach is
  **not** an error — it reconciles to `status="breached"`. `settle_contract` is the pure builder.
- **Reconciliation across the boundary.** `reconcile(a, b)` ties two independently-produced records
  out into a `Reconciliation` (`.agrees`, `.hashes_match`, `.discrepancies`) — a disagreement is
  pinpointed as a dispute, not merely flagged.
- **The settlement book.** `app.use_settlement_book()` attaches a `SettlementBook` — an org's
  durable, **hash-chained** ledger of settlements (the analogue of the `SagaJournal`). `.settle(...)`
  reconciles, signs as the owner's side, links the record into the chain, audits the verdict (the
  `settlement` action), and closes the reputation loop; `book.verify(verifier=)` recomputes the whole
  ledger offline and pinpoints any tampered record (`broken_at`); `app.settlement_report(
  counterparty=)` rolls the books up per counterparty beside the cost report. A unique `book_id` by
  default (no cross-store collision); pass a stable one to resume across restarts.
- **Settling a whole saga.** `app.settle_saga(result, *, contracts={id: Contract})` / `settle_saga`
  meters each contracted step from the durable journal and reconciles the per-step delivery against
  the matching contract, appending one signed record per contract.
- **Reputation-closing.** A settled overrun or shortfall debits the seller on an attached
  `ReputationLedger`, so reliability earned in delivery weights the next negotiation — bounded and
  reversible, never singled out.
- **New error** `SettlementError` (`SETTLEMENT_ERROR`) with an error-catalog entry; a `settlement`
  VincioBench family with two SLOs (metering accuracy; settlement integrity), companion budgets, and
  [`examples/69_agent_to_agent_settlement.py`](examples/69_agent_to_agent_settlement.py); a
  [settlement guide](docs/guides/settlement.md).

## [3.24.0] - 2026-06-22

Cross-org workflow choreography. With agents that discover, negotiate, and contract across
organizations, this release adds the next rung: the **durable work** they coordinate — a
long-running, compensating workflow that spans more than one organization's agent fabric, the
choreography analogue of the in-process durable graph, now crossing trust boundaries. Each org
governs and audits its own steps on its own hash-chained chain; only a typed contract and audited
handoffs cross a trust boundary; and a failure on one side triggers deterministic compensation
across the whole choreography. Landed in the *same* governed, audited, budgeted runtime, never as a
hosted control plane. Entirely additive and backward-compatible — `API_VERSION` stays `3.0`, and
the whole theme runs offline against deterministic local participants or in-process over the A2A
fabric.

### Added

- **The choreography package (`vincio.choreography`).** A `Saga(name=).step(name, *, participant=,
  action=, compensation=, payload=, build=, contract=, retries=)` defines an ordered, compensating
  cross-org workflow. `app.choreograph(saga, *, participants=, input=, saga_id=, interrupt_after=)`
  / `achoreograph` drives it with a `Choreography` engine and returns a `SagaResult` (`.status`
  completed|compensated|failed|interrupted, `.completed_steps`, `.compensated_steps`, `.failed_step`,
  `.output` / `.output_of(step)`, `.journal`). `participants` maps an org id to a `Participant` — a
  `RemoteParticipant` over A2A or, as a convenience, a dict of `{action: handler}` callables wrapped
  in a `LocalParticipant`; a handler returns a dict (output) or a `StepOutcome` declaring delivered
  `cost_usd` / `latency_ms` / `quality`. A later step's payload can be derived from prior steps'
  outputs with a `build` callable over a `SagaContext`.
- **Per-org governance, no shared control plane.** The coordinator audits each dispatched
  `StepRequest` handoff on its own hash-chained chain (the `choreography_step` action) while each
  participant audits its execution on its own — only the typed contract and the audited handoff
  cross a trust boundary.
- **Durable & resumable.** The `SagaJournal` is checkpointed to the metadata store (kind
  `choreography_sagas`) after every step, so `app.resume_choreography(saga, saga_id, *,
  participants=)` resumes after a restart on a fresh engine and never re-runs a completed step;
  `interrupt_after` cooperatively pauses a long saga into a resumable state. The journal is
  **hash-chained**, so `journal.verify(verifier=)` recomputes it **offline** and pinpoints any
  tampered record (`broken_at`); an optional engine `signer` signs each record.
- **Compensating saga.** A forward step that returns `ok=False`, raises, or **breaches its step
  `Contract`** (delivered cost/latency/quality checked against the agreed terms) triggers
  deterministic compensation of the completed steps in **reverse order** (a compensation handler
  receives the forward output under `payload["forward_output"]`). A clean unwind is
  `status="compensated"`; a compensation that itself fails ends `status="failed"` (or raises
  `CompensationError` with `raise_on_compensation_failure=True`).
- **Over the A2A fabric.** `app.serve_choreography(handlers, *, org_id=)` exposes an org's handlers
  as an A2A agent (a `choreograph` skill, audited on that org's chain), and `RemoteParticipant(
  client, org_id=)` dispatches a step to a remote org over A2A byte-for-byte the same as a local
  participant.
- **New errors** `ChoreographyError` (`CHOREOGRAPHY_ERROR`) and `CompensationError`
  (`COMPENSATION_FAILED`) with error-catalog entries; a `choreography` VincioBench family with two
  SLOs (saga durability survives a restart; a failure compensates in reverse order), companion
  budgets, and [`examples/68_cross_org_workflow_choreography.py`](examples/68_cross_org_workflow_choreography.py);
  a [choreography guide](docs/guides/choreography.md).

## [3.23.0] - 2026-06-22

Agent negotiation & contracting. Vincio already governs a fabric of agents over A2A and the
MCP registry behind an allow-list, scores per-member reliability with a reputation ledger, and
discounts an unreliable member's pull on a federated round. This release adds the next rung:
**bounded negotiation and contracting** between agents in a multi-org crew — a buyer agent and a
seller agent converge on a price/SLA/scope contract under a hard budget, the contract is a typed,
signed, audited artifact both sides verify offline, and the counterparty's reputation weights the
deal. Landed in the *same* governed, audited, budgeted runtime, never as a hosted marketplace.
Entirely additive and backward-compatible — `API_VERSION` stays `3.0`, and the whole theme runs
offline against deterministic local parties or in-process over the A2A fabric.

### Added

- **The negotiation package (`vincio.negotiation`).** A `Negotiation` runs a typed
  alternating-offers bargain between a buyer and a seller `Party` (each a `NegotiationPosition`
  with per-issue ideal/reservation preferences and a time-dependent concession curve, run as a
  deterministic `LocalParty`). `app.negotiate(scope, *, buyer=, seller=, budget=NegotiationBudget(
  max_rounds=, deadline_s=))` / `anegotiate` returns a `NegotiationResult` (`.status`, `.agreed`,
  `.contract`, `.rounds`, `.offers`, `.deadline_hit`). **Termination is guaranteed**: a deal when
  the parties' acceptable regions overlap (`AC_next` acceptance), a clean no-deal when they do not,
  a partial result on a wall-clock deadline. `buyer_position` / `seller_position` build positions;
  `IssuePreference` / `Offer` are the typed primitives.
- **Typed, signed, offline-verifiable contracts.** On agreement a `Contract` (`ContractTerms` over
  price / SLA / scope / quality) is minted and **signed by both parties** (with an explicit signer,
  the audit-chain signer, or a per-app key via `app.contract_signer`). `contract.verify(signer)`
  recomputes the content hash and checks every signature **offline from the bytes alone** (a
  tampered term or forged signature → invalid); `contract.to_budget()` lowers price→`max_cost_usd`
  / SLA→`max_latency_ms`, and `contract.check(...)` / `app.enforce_contract(...)` detect a breach,
  so the orchestrator enforces a contract **like any other budget**. The outcome and the signed
  contract land on the hash-chained audit log (`negotiation` / `contract_signed` /
  `contract_fulfillment`).
- **Reputation-weighted offers.** When a `ReputationLedger` is attached
  (`app.use_reputation_ledger()`), a local party discounts a counterparty's offers by its
  reputation weight (`[floor, 1]`, bounded and reversible — discounted, never singled out or
  zeroed); `select_offer(results, buyer_position, reputation=)` picks the reputation-weighted best
  deal among competing sellers, and `app.enforce_contract` debits a breaching seller, closing the
  loop from delivery back to reputation.
- **Over the A2A fabric.** `app.serve_negotiation(party)` exposes a local `Party` as an A2A agent
  (a `negotiate` skill), and `A2ANegotiator(client, member_id=, role=)` drives a remote counterparty
  over A2A byte-for-byte the same as a local one — the remote party's identity is pinned to the
  directory-resolved member id, never the self-asserted one on the wire.
- **New errors** `NegotiationError` (`NEGOTIATION_ERROR`) and `ContractError` (`CONTRACT_VIOLATION`)
  with error-catalog entries; a `negotiation` VincioBench family with two SLOs (negotiation
  terminates within budget; contract integrity verifies offline), companion budgets, and
  [`examples/67_agent_negotiation_and_contracting.py`](examples/67_agent_negotiation_and_contracting.py);
  a [negotiation guide](docs/guides/negotiation.md).

## [3.22.0] - 2026-06-22

MCP Apps & the evolving MCP spec. Vincio already speaks MCP in-process — client and
server, tools through the permissioned runtime, resources as cited evidence — and streams a
run as AG-UI generative-UI events. This release adopts the spec's newer surface and lands it
in the *same* governed, audited, budgeted runtime, never as a hosted service: server-rendered
UI (MCP Apps) surfaced through the existing AG-UI channel, a typed elicitation request gated
by the same approval + rail machinery a write tool passes, and evolving-spec parity (protocol
negotiation + a stateless-core transport mode). Entirely additive and backward-compatible —
`API_VERSION` stays `3.0`, the established MCP client/server paths are unchanged, and the whole
theme runs offline with a deterministic in-process server substitute.

### Added

- **MCP Apps (server-rendered UI).** `app.mcp_app(name, max_render_tokens=)` returns an
  `MCPAppBridge` (`vincio.mcp`) that reads a consumed server's `ui://` UI resources and lowers
  each into an AG-UI `CUSTOM` `mcp.ui` event (`vincio.server.agui.mcp_ui_event`). Each render is
  governed: untrusted-external provenance, token-metered against the run (an oversized render is
  refused, no event emitted), and recorded on the hash-chained audit log (`mcp_ui_render`).
  `bridge.to_agui_events()` / `bridge.stream(base)` (splices UI before `RUN_FINISHED`). A tool
  result may embed a UI resource — surface text + `[MCPUIResource]` with `client.call_tool_ui`,
  or return an `MCPUIResource` from an app tool. `is_ui_resource` / `MCPUIRender`.
- **Elicitation (governed mid-call input).** `ElicitationGate` / `ElicitationRequest` /
  `ElicitationResponse` / `ElicitationPolicy` / `ElicitationDecision` / `ElicitationAction`
  (`vincio.mcp`). `app.add_mcp_server(..., elicitation=collector, elicitation_approval=fn,
  elicitation_policy=...)` routes a server's `elicitation/create` through the gate: an approver
  may deny the request, the collected value is screened through the input `RailEngine` (a secret
  or injection value is declined), and an accepted value is wrapped `TaintedValue.untrusted(...)`
  (`mcp:<server>:elicitation`) so it is contained like any other untrusted input. Every decision
  is audited (`mcp_elicit`). A served app initiates one with `MCPServer.elicit(message, schema=)`.
- **Evolving-spec parity.** Protocol-version negotiation (`negotiate_version`,
  `SUPPORTED_PROTOCOL_VERSIONS`) honours a peer pinned to an older stable revision and is recorded
  on both the client (`negotiated_version`) and server; `StreamableHTTPTransport(stateless=True)`
  is the stateless-core transport mode (no `Mcp-Session-Id`). A `resource_content` JSON-RPC helper
  for embedded-resource content blocks.
- **A `mcp_apps` VincioBench family** with three SLOs (UI governed through AG-UI; elicitation
  contained by approval + rails; spec-revision negotiated), companion budgets, and
  [`examples/66_mcp_apps_and_elicitation.py`](examples/66_mcp_apps_and_elicitation.py).

## [3.21.0] - 2026-06-22

Edge / WASM in-process runtime. Vincio's promise is "runs in your process" — and the
dependency-free core (the prompt and context compilers, the vectorized scorer with its
pure-Python fallback, the deterministic rails, and the offline-first evidence path) already
has no native dependencies on the default path. This release takes that core to the edge:
the same **compile → score → rail → pack** pipeline runs in a browser (Pyodide/WASM) or an
edge worker, behind a thin in-process boundary, bounded by an edge profile — not as a fork,
but as the same library under a build target. Entirely additive and backward-compatible —
`API_VERSION` stays `3.0`, the server path is unchanged and remains the default, and the
whole theme runs offline (no provider, store, network, or filesystem).

### Added

- **The edge runtime (`vincio.edge`).** `EdgeRuntime` turns an `EdgeRequest` (a task,
  instructions, constraints, evidence, and memory — all the platform's own typed inputs)
  into an `EdgeResult` (a bounded, slim `ContextPacket`, the rendered model-ready prompt,
  the merged input/output rail outcome, and the measured resident footprint and latency)
  with no model call, network hop, filesystem, or caller-owned event loop. `.run` is
  synchronous (works under a WASM host's loop); `.arun` is async; a plain string is accepted
  for the common case. It is parity by construction — the runtime *delegates* to the
  canonical `ContextCompiler` and `RailEngine`, never re-implementing them.
- **The bounded edge profile.** `EdgeProfile` caps the compiled packet's resident footprint,
  token window, and evidence/memory counts for a constrained target, and lowers directly to
  the *same* `ContextCompilerOptions` the server compiler reads (`.to_compiler_options()`).
  Presets: `EdgeProfile.browser()` (256 KiB / 4096 tok), `.worker()` (the default), and
  `.server_like()` (for parity testing). The footprint stays under the cap as the candidate
  corpus grows 10×, held by the same slimming + eviction the server's resident-memory budget
  uses; `run(..., strict=True)` raises `EdgeError` instead of reporting `within_profile=False`.
- **Parity, not a fork.** `verify_edge_parity()` compiles the same inputs through the edge
  runtime and through a direct server `ContextCompiler` under the same profile and asserts a
  byte-identical packet (`spec_hash`, evidence selection, token count), plus that the runtime
  delegates to the canonical compiler/rail engine. `edge_manifest()` statically scans every
  module on the compile/score/rail/pack path and certifies it imports nothing native or
  optional unconditionally (NumPy stays behind its guarded pure-Python fallback) — the
  WASM-buildability guarantee.
- **Host detection & app surface.** `edge_environment()` / `is_wasm_runtime()` detect a
  Pyodide/WASI host without executing anything; `app.edge_runtime(profile=None)` builds a
  runtime seeded with the app's rails so the edge path enforces the same deterministic safety
  the server does (output rails screen the rendered context, refusing a secret that leaked
  from evidence into the prompt). New error `EdgeError` (`EDGE_ERROR`, catalogued).
- **`edge` VincioBench family + 3 SLOs.** Holds byte-identical parity, the bounded resident
  profile under a 10× corpus (eviction firing under load), the no-native-imports certificate,
  rails enforced at the edge, and fully-offline operation. SLOs: `edge_parity_byte_identical`,
  `edge_bounded_profile`, `edge_core_no_native_imports`.
- **Example.** `examples/65_edge_wasm_runtime.py` and the [edge guide](docs/guides/edge.md) —
  a fully offline walkthrough of compiling at the edge, the bounded profile under load, rails
  at the edge, parity, and host detection.

### Changed

- The public surface gains `EdgeRuntime`, `EdgeRequest`, `EdgeResult`, `EdgeProfile`,
  `EdgeEnvironment`, `EdgeManifest`, `EdgeParityReport`, `edge_environment`,
  `is_wasm_runtime`, `edge_manifest`, and `verify_edge_parity`; `vincio.core.errors` gains
  `EdgeError`; `ContextApp` gains `edge_runtime()`. The next scheduled roadmap theme is
  **MCP Apps & the evolving MCP spec** (target 3.22).

## [3.20.0] - 2026-06-22

Native video understanding & generation. The multimodal packet already scores, budgets,
orders, and cites image and table evidence beside text, and generation flows images and
audio **out** with C2PA provenance. Video was the modality not yet first-class — a recorded
meeting, a screen capture, a product demo reduced to a transcript or a handful of stills,
losing the temporal structure that makes it evidence. This release makes video first-class
on the **existing** packet, never a new plane: a typed video reference and content part,
deterministic frame sampling and temporal segmentation, a video analyzer that lowers a clip
into typed evidence the context compiler scores and cites beside everything else, temporal
grounding that carries a segment's time range through to the citation, and C2PA-bound video
generation/editing on the same metered, audited path as images and audio. Entirely additive
and backward-compatible — `API_VERSION` stays `3.0`, the dependency-free offline path is the
default (a deterministic mock substitutes for every model/codec call), and the real
frame-decode path installs behind the new `vincio[video]` extra.

### Added

- **Video as a typed modality (`vincio.core`).** A `VideoRef` (path/url, media type,
  duration, fps, detail) joins `ImageRef` / `AudioRef`; `ContentPart` gains a `video` part
  and `UserInput` a `video` list. `EvidenceItem` gains `modality="video"`, a `video` carrier,
  and a `time_range` temporal locator whose `citation_ref` renders `<source>:t<start>-<end>`;
  `scorable_text` and the modality-aware token cost cover video. `core.media` gains
  `encode_video_bytes` and `DEFAULT_MAX_VIDEO_BYTES`.
- **Video understanding (`vincio.documents.video`).** Deterministic, dependency-free
  `sample_frame_times` (frame sampling) and `segment_timeline` (temporal segmentation)
  address a clip without decoding it. A `VideoAnalyzer` turns a clip into a `VideoAnalysis`
  (a `VideoSegment` timeline of transcripts/captions and sampled `VideoFrame`s);
  `MockVideoAnalyzer` keeps offline runs deterministic, and `ProviderVideoAnalyzer` +
  `PyAVFrameExtractor` decode and caption frames behind the `vincio[video]` extra.
  `video_evidence_items` lowers an analysis into typed, time-stamped, citable evidence.
- **First-class in the packet & temporal grounding.** The context compiler scores, budgets,
  orders, and cites video evidence beside text/image/table (the packet serializes the video
  payload and its `time_range`). Retrieval chunking carries a transcript segment's
  `(start, end)` onto the chunk and `_to_evidence`, and the cited-report builder resolves a
  claim to a `time_range`, rendering the footnote at the moment (`, t10–15s`) — so a
  video-grounded answer is auditable at sub-clip resolution. The evidence compressor now
  only compresses text, so a media item's footprint is never undercounted.
- **Video generation with provenance (`vincio.generation.video`).** A `VideoProvider`
  surface — `generate_video` / `edit_video` — over a deterministic `MockVideoProvider`,
  OpenAI Sora (`OpenAIVideoProvider`), Google Veo (`GoogleVideoProvider`), and a generic
  `HTTPVideoProvider`. Every clip carries a C2PA `ProvenanceManifest` bound to its bytes
  (`video_cost` / `VideoPrice` price it); editing marks the manifest synthetic-and-edited.
- **App surface.** `app.load_video(path, *, analyzer)` ingests a clip as a
  temporally-segmented document; `app.generate_video` / `app.aedit_video` (+ sync wrappers)
  generate/edit video metered against the budget, audited (`video_generate` / `video_edit`),
  and C2PA-stamped — the same choke point images and audio use.
- **`video` VincioBench family + 3 SLOs.** Holds deterministic sampling, full-timeline
  segmentation, video as a first-class compiler candidate, temporal-grounding accuracy with
  the timestamp surviving into the citation, and provenance binding (tamper rejected, edit
  marked). SLOs: `video_temporal_grounding`, `video_generation_provenance_bound`,
  `video_first_class_evidence`.
- **Example.** `examples/64_video_understanding_and_generation.py` and the
  [video guide](docs/guides/video.md) — a fully offline walkthrough of understanding a clip,
  citing it at the moment, and generating provenance-bound video.

### Changed

- The public surface gains `VideoProvider`, `VideoGenRequest`, and `MockVideoProvider`;
  `vincio.generation` additionally exports the video providers, `VideoGenResponse`,
  `GeneratedVideo`, and `video_cost`; `vincio.documents` exports the video analyzers,
  `VideoAnalysis` / `VideoSegment` / `VideoFrame`, `sample_frame_times`, `segment_timeline`,
  `video_evidence_items`, and `load_video`; `vincio.core.types` gains `VideoRef`.
  `MediaGenerationError` now also covers video. A new `vincio[video]` extra installs the
  real frame-decode backend (PyAV + Pillow). The next scheduled roadmap theme is **edge /
  WASM in-process runtime** (target 3.21).

## [3.19.0] - 2026-06-22

Formal verification of governance invariants. The platform already **enforces** its
governance invariants at runtime — residency refuses an out-of-region egress, provable
erasure binds a signed proof to the removed-id set, the budget caps spend, and the
injection-containment gate stops an untrusted-tainted argument reaching a side-effecting
tool without a user-minted capability — and records each decision on the signed audit
chain. What was not yet first-class is a **machine-checkable proof that those invariants
hold across the whole input space, ahead of any single run** — a property checked by
construction rather than observed after the fact. This release adds it: a deterministic,
in-process verifier that proves four governance invariants over their whole bounded,
typed state space by exhaustive bounded model checking, yields a minimal counterexample
on a violation, and records the content-hashed verdict on the hash-chained audit log.
Entirely additive and backward-compatible — `API_VERSION` stays `3.0`, the
dependency-free offline path is the default, and verification is **opt-in** (nothing
runs unless you call `app.verify_governance()`).

### Added

- **The verifier (`vincio.governance.verification`).** A `GovernanceVerifier` checks a
  list of `Invariant`s — each a formal specification, a tuple of `StateVariable`s, and a
  predicate over an assignment — by enumerating the *full* Cartesian product of the
  variables' representative values. A `held=True` verdict means the predicate was
  confirmed at every point of the bounded domain (`states_checked == domain_size`) — a
  proof over the modeled domain, not a sample. `verify()` returns a content-hashed
  `VerificationReport` (`held`, per-invariant `InvariantResult`s, `content_sha256`,
  reproducible via `report.verify()`).
- **The four platform invariants, bound to the shipped machinery.** `containment_invariant`
  proves `untrusted ⇒ no unapproved capability` against the *same* gate the
  `DualPlaneExecutor` runs (the extracted `requires_authority` predicate) vs the
  `ContainmentEvent.is_escalation` specification; `residency_invariant` proves an enforced
  `ResidencyPolicy` admits egress only to an in-jurisdiction region (and refuses an unknown
  one fail-closed); `budget_invariant` proves the canonical hard-cap predicate
  (`within_budget`, behind the dollar/energy/carbon caps) never admits an overspend;
  `erasure_invariant` proves `verify_erasure_proof` accepts a proof iff its removed-id set
  is intact. `default_invariants()` returns the four, fail-closed.
- **Counterexample, not just a verdict.** A failed property returns a delta-minimized
  `Counterexample` — the concrete violating assignment (the input, the labels, the
  capability gap), with each variable relaxed back toward its benign default while the
  violation persists — rendered one-line via `.render()`.
- **Auditable & offline.** No external prover service is consulted; the verdict lands on
  the hash-chained, verifiable audit log as a `governance_verification` decision (`allow`
  when held, `deny` otherwise), carrying each invariant's verdict and any counterexample.
- **App surface.** `app.verify_governance(invariants=None, *, record=True,
  raise_on_violation=False)` runs the four invariants — the residency one reflecting the
  app's own `deny_on_unknown` posture, so a fail-open configuration is caught — records the
  verdict, and returns the report; `raise_on_violation=True` raises the new
  `GovernanceVerificationError` (`GOVERNANCE_INVARIANT_VIOLATED`, under `GovernanceError`)
  carrying the counterexamples.
- **`verification` VincioBench family + 3 SLOs.** Holds the property-holds,
  proof-not-sample, four-invariant-coverage, counterexample-on-violation (residency,
  budget, and containment), minimal-counterexample, deterministic, and auditable-offline
  invariants. SLOs: `governance_invariants_proven`,
  `governance_counterexample_on_violation`, `governance_verification_auditable_offline`.
- **Example.** `examples/63_governance_invariant_verification.py` and the
  [verification guide](docs/guides/governance-verification.md) — a fully offline
  walkthrough of proving the invariants, the proof-not-sample property, the
  counterexample on a fail-open posture and a buggy budget cap, and the audit trail.

### Changed

- The public surface gains `GovernanceVerifier`, `VerificationReport`, `InvariantResult`,
  `Counterexample`, and `Invariant`; `vincio.governance` additionally exports
  `StateVariable`, the four invariant builders, `default_invariants`, and `within_budget`;
  `vincio.core.errors` gains `GovernanceVerificationError`. `vincio.security` gains the
  shared `requires_authority` gate predicate (and `AUTHORIZED`); the `DualPlaneExecutor`
  now gates on it, so the runtime guard and the proof share one source of truth (no
  behavior change). The next scheduled roadmap theme is **native video understanding &
  generation** (target 3.20).

## [3.18.0] - 2026-06-22

Energy & carbon accounting. The cost report already makes a run's dollar spend an
auditable number held by a budget SLO, and the resident-memory budget does the same
for footprint — but the platform reported **nothing about a run's energy or carbon**,
the disclosure sustainability-reporting regimes are beginning to demand. This release
adds the missing rung — a per-run **energy** (watt-hours) and estimated **carbon**
(grams CO₂e) figure on the *existing cost-report surface*, the energy analogue of the
dollar budget, never a new plane. Entirely additive and backward-compatible —
`API_VERSION` stays `3.0`, the dependency-free offline path is the default, accounting
is **off until explicitly enabled**, and the estimate is computed in-process from a
deterministic intensity table with no external service.

### Added

- **The energy/carbon estimation model (`vincio.observability.energy`).** An
  `EnergyIntensityTable` (the energy analogue of the `PriceTable`) maps a model to an
  `EnergyProfile` — watt-hours per million input/output tokens, seeded from the
  `ModelRegistry` by tier (decode dominates prefill; a stronger tier draws more) and
  overridable per model — scales the result by a datacenter `pue`, and multiplies by a
  per-region grid carbon factor (g CO₂e/kWh) from a built-in `DEFAULT_CARBON_INTENSITY`
  table (overridable per region). `estimate(model, usage, region=)` returns a decomposed
  `EnergyEstimate` (`energy_wh`, `co2e_grams`, the input/output breakdown, the resolved
  region and intensity). The estimate is mechanical and reproducible.
- **On the cost-report surface.** `CostTracker` accrues `energy_wh` / `co2e_grams`
  (surfaced in `summary()`), each attributed `CostEvent` / `CostRow` carries its energy
  and carbon, and `RunResult` gains `energy_wh` / `co2e_grams`. `CostLedger` gains
  `total_energy` / `total_co2e` and an `energy_report(by=...)` returning an
  `EnergyReport` / `EnergyRow` — rolled up from the *same* attributed events the cost
  report uses, by tenant / feature / user / model / provider / run.
- **Budgeted like a dollar.** An `EnergyBudget` (energy and/or carbon ceiling, scoped,
  rolling period) added to the `BudgetManager`; `check_energy(...)` returns an
  `EnergyBudgetDecision`, and a run whose scope has accrued past the envelope is
  **refused** on the same audit path as a hard cost cap — an `energy_budget` audit entry
  and an `energy.budget_exceeded` event.
- **Auditable & offline.** No external service is consulted; both the per-run estimate
  (on the terminal `run` audit entry, when enabled) and every refusal land on the
  hash-chained, verifiable audit log.
- **App surface.** `app.use_energy_accounting(region=, pue=, carbon_intensity=)` turns
  accounting on (off by default) and pins the deployment region / overhead / grid
  factors; `app.set_energy_budget(scope=, id=, limit_wh=, limit_co2e_grams=, period=)`
  adds an envelope (enabling accounting on first use); `app.energy_report(by=)` rolls up
  the estimate next to `cost_report`. `EnergyBudgetError` (`ENERGY_BUDGET_INVALID`,
  under `ObservabilityError`) guards a budget set with no ceiling.
- **`energy` VincioBench family + 3 SLOs.** Holds the per-run-estimate,
  budget-refused, auditable-offline, decode-dominates, tier-monotonic,
  region-intensity-differs, carbon-tracks-energy, off-by-default, and on-cost-surface
  invariants. SLOs: `energy_per_run_estimate`, `energy_budget_refusal`,
  `energy_auditable_offline`.
- **Example.** `examples/62_energy_carbon_accounting.py` — a fully offline walkthrough
  of enabling accounting, the per-run estimate, the region-dependent carbon, the cost
  surface roll-up, the budget refusal, and the audit trail.

### Changed

- The public surface gains `EnergyProfile`, `EnergyEstimate`, `EnergyIntensityTable`,
  `EnergyBudget`, and `EnergyReport`; `vincio.observability` additionally exports
  `default_energy_table`, `DEFAULT_CARBON_INTENSITY`, `EnergyRow`, and
  `EnergyBudgetDecision`; `vincio.core.errors` gains `EnergyBudgetError`. `CostTracker`,
  `CostEvent`, `CostRow`, `CostLedger`, `BudgetManager`, and `RunResult` gain
  backward-compatible energy/carbon fields and methods (all defaulting to the
  pre-accounting behavior — zero until enabled). The next scheduled roadmap theme is
  **formal verification of governance invariants** (target 3.19).

## [3.17.0] - 2026-06-22

Cross-fleet reputation & weighting. The federated round merged every member's
contribution with **equal weight**, and the privacy accountant bounds what each
member can *leak* — but the platform had **no notion of a member's track record**: a
member whose contributions repeatedly fail the no-regression gate still pulled the
shared consensus geometry as hard as one whose contributions consistently help. This
release adds the missing rung — a per-member **reputation**, earned only from how each
contribution fared against the gate (never from raw traffic), that discounts an
unreliable or adversarial member's pull on the consensus. Entirely additive and
backward-compatible — `API_VERSION` stays `3.0`, the dependency-free offline path is
the default, and without a ledger the federated round behaves exactly as before.

### Added

- **`ReputationLedger` + the reputation model (`vincio.optimize.reputation`).** A
  per-member reliability signal kept as a Beta-Bernoulli posterior over no-regression
  gate outcomes — a robust generalization of the existing `successes / calls`
  reliability scoring: a newcomer earns the benefit of the doubt from a configurable
  prior, a repeatedly-regressing member decays toward a floor, and (with `decay < 1`) a
  reformed member recovers. `reputation(member)` is the posterior mean, `weight(member)`
  maps it to an aggregation weight in `[weight_floor, 1]`, `record_outcome(member,
  passed=)` composes one verdict, `record_round(members, passed=)` credits a whole
  round, and `assign(members)` produces the round's weight vector.
- **`ReputationConfig` / `MemberReputation` / `ReputationWeights` / `ReputationReport` /
  `ReputationRow`.** The configuration (prior pseudo-counts, decay, weight band),
  per-member snapshot, per-round weight assignment, and the per-member roll-up.
  `ReputationError` inherits `OptimizationError`'s `OPTIMIZATION_ERROR` code (no new
  catalog entry).
- **Reliability-weighted aggregation.** `SecureAggregator(reputation=ledger)` (or an
  explicit `weights=` map) weights a member's contribution by its reputation before
  distilling the consensus subspace, so a regressor is discounted **without being
  singled out**. The weight is folded into the contribution *before* the
  secure-aggregation masks (via `ContributionBuilder.build(..., reputation_weight=)`),
  so the masks still cancel exactly; the aggregator refuses to re-weight an
  already-masked contribution (`FederatedError`), surfacing the cryptographic
  constraint rather than silently corrupting the merge. `Contribution` carries an
  auditable `reputation_weight`; the merged subspace records the per-member weights.
- **Wired into the gated round.** When a ledger is bound,
  `app.federated_improvement` / `app.adopt_federated` weight each member's contribution
  and record the round's gate verdict back to the ledger (`FederatedPolicy.record_reputation`,
  default on); `FederatedRoundResult` carries the applied `reputation_weights`. The
  discount is bounded and reversible — a weight only ever lowers a member's pull, and
  adoption still clears the same no-regression and canary gates — so reputation can
  never bypass the quality bar.
- **Audit-chain reputation.** Every update lands on the hash-chained, verifiable audit
  log (`reputation_update`), and `ReputationLedger.from_audit(audit)` /
  `replay_from_audit` reconstruct the whole ledger from the chain alone — a member's
  standing is a mechanical, replayable number.
- **App surface.** `app.use_reputation_ledger(config=)` attaches a ledger wired to the
  audit chain, event bus, and store; `app.reputation_report(member=)` rolls up each
  member's score and weight next to the cost and privacy reports.
- **`reputation` VincioBench family + 2 SLOs.** Holds the discount-the-regressor,
  weight-bounded/floored, audit-replayable, adopts-at-least-as-good, and
  gate-not-bypassed invariants. SLOs: `reputation_discount_the_regressor`,
  `reputation_no_regression`.
- **Example.** `examples/61_cross_fleet_reputation_weighting.py` — a fully offline
  walkthrough of earning, weighting, the discount, the bounded/reversible gate, and the
  audit replay.

### Changed

- The public surface gains `ReputationLedger`, `ReputationConfig`, `MemberReputation`,
  `ReputationReport`, and `ReputationError`; `vincio.optimize` additionally exports
  `ReputationWeights` and `ReputationRow`. `SecureAggregator`, `ContributionBuilder`,
  `FederatedImprovement`, `FederatedPolicy`, `FederatedRoundResult`, and `Contribution`
  gain backward-compatible reputation fields/parameters (all defaulting to the
  unweighted behavior).

## [3.16.0] - 2026-06-22

Differential-privacy memory & training. The federated round bounds a *single
member's per-round influence* with clipping and an optional Gaussian mechanism, but
the platform had **no end-to-end privacy accountant**: a per-subject, cross-round
budget that composes every memory consolidation and learning round a subject's data
touches and *refuses* once the budget is spent. This release adds it — a provable,
composing, per-subject privacy budget over memory consolidation and the whole
learning loop. Entirely additive and backward-compatible — `API_VERSION` stays
`3.0`, the dependency-free offline path is the default, and nothing below runs unless
you opt in.

### Added

- **`PrivacyAccountant` + the accountant's math (`vincio.governance.privacy`).** A
  Rényi / moments accountant that composes the cumulative `(ε, δ)` a subject's data
  has spent across every accounted release into one running budget — far more tightly
  than naively summing each step's `ε`. `gaussian_rdp(z, sample_rate=, steps=)` is
  the (Poisson-sub-sampled) Gaussian-mechanism RDP curve (exact `α / 2z²` at full
  batch, the moments-accountant binomial bound under sub-sampling), and
  `rdp_to_epsilon(rdp, delta=)` is the standard RDP→`(ε, δ)` conversion.
- **`PrivacyMechanism` / `PrivacyBudget` / `PrivacySpend` / `PrivacyDecision`.** A
  mechanism models one Gaussian release (noise multiplier, sample rate, steps); a
  budget is a per-subject (or default) `(ε, δ)` ceiling with an `on_breach` policy
  (`refuse` — a hard cap — or `downweight` — clip harder so the release's sensitivity
  and privacy cost fit). `check` decides whether a release fits; `charge` gates and
  commits, raising `PrivacyBudgetError` (code `PRIVACY_BUDGET_EXCEEDED`) on a refusal.
  Budgets are per-subject and isolated.
- **`PrivacyReport` + `app.privacy_report()`.** A per-subject roll-up of `ε` spent
  against the ceiling, with operation and refusal counts — the privacy analogue of
  `app.cost_report()`. Every spend (`privacy_spend`) and refusal (`privacy_refused`)
  lands on the hash-chained, verifiable audit log.
- **App surface.** `app.use_privacy_accountant(default_budget=, default_mechanism=)`
  attaches an accountant wired to the audit chain and store;
  `app.set_privacy_budget(subject_id=, epsilon=, delta=, on_breach=)` is the
  one-liner for a budget.
- **Wired integrations.** Memory consolidation (`app.memory.consolidate(session_id,
  user_id=)`) charges the subject's budget and refuses an over-budget consolidation —
  the `ConsolidationReport` now carries `privacy_refused` and `privacy_epsilon`.
  Federated contributions compose the **same** budget when the federated
  `PrivacyConfig` configures the Gaussian mechanism (`dp_epsilon` set); an
  over-budget contribution is refused, a down-weighted one is released more privately
  (the mechanism's `ε` scaled down — more noise relative to sensitivity).
- **`privacy` VincioBench family + 3 SLOs.** Holds the Gaussian-RDP exactness,
  cross-round composition, budget refusal, per-subject isolation, down-weight,
  memory- and federated-gating, report, and audit-chain invariants. SLOs:
  `privacy_budget_composes`, `privacy_budget_refuses`, `privacy_budget_auditable`.
- **Example.** `examples/60_differential_privacy_memory_training.py` — a fully
  offline walkthrough of accounting, refusal, federation, down-weight, and the report.

### Changed

- `MemoryEngine` accepts an optional `privacy_accountant` / `privacy_mechanism`;
  consolidation gates on the subject's budget when one is attached (unaccounted and
  unchanged otherwise).
- The public surface gains `PrivacyAccountant`, `PrivacyBudget`, `PrivacyMechanism`,
  `PrivacySpend`, `PrivacyDecision`, `PrivacyReport`, and `PrivacyBudgetError`;
  `vincio.governance` additionally exports `PrivacyRow`, `gaussian_rdp`, and
  `rdp_to_epsilon`.

## [3.15.0] - 2026-06-22

Federated / cross-org self-improvement. The platform already learns from its own
traffic three ways — the on-policy RLVR loop, the distillation flywheel, and
on-device local adaptation — but always *within one trust boundary*. This release
adds the rung above them: **sharing what was learned across organizations without
sharing the raw traffic**, so a fleet of members improves together while each
member's data stays put. Entirely additive and backward-compatible — `API_VERSION`
stays `3.0`, the dependency-free offline path is the default, and nothing below runs
unless you opt in.

### Added

- **`Contribution` + `ContributionBuilder` (`vincio.optimize.federated`).** A
  member's privacy-preserving federated update: the `d×d` weighted *scatter* of its
  local prompt-embedding subspace — a second-moment sufficient statistic from which
  no individual prompt or response is recoverable — and nothing else. The builder
  embeds the member's prompts, forms the scatter, **clips** it to a sensitivity
  bound, optionally adds the **differential-privacy** Gaussian mechanism, and folds
  in **secure-aggregation** masks. The wire object carries no raw traffic, plus a
  consent attestation and a residency tag.
- **`PrivacyConfig`.** The opt-in privacy posture: `clip_norm` bounds a member's
  sensitivity, `dp_epsilon`/`dp_delta` parameterize the Gaussian mechanism
  (`noise_sigma()`), `secure_aggregation` toggles the cancelling masks, and
  `min_contributors` is the round-level k-anonymity floor. `seed` keeps noise and
  masks reproducible offline.
- **`SecureAggregator` + `FederatedSubspace`.** Sums the masked contributions — the
  pairwise masks cancel across the exact participant set, so the aggregator recovers
  the fleet scatter without ever observing an individual update — refuses a round
  below `min_contributors` or one mixing base models, embedding dimensions, or
  disallowed residency regions, and extracts the consensus subspace by deterministic
  federated PCA (top eigenvectors of the aggregate scatter, via power iteration with
  deflation). `subspace.digest` is the behaviour-tracking content address.
- **`refit_with_subspace`.** Re-fits a member's **own** `LocalAdapter` against the
  shared subspace: the geometry is the fleet's consensus, the codes and grounded
  targets are the member's own local data — so adoption imports the fleet's learned
  structure without importing anyone's text. The result is an ordinary `LocalAdapter`
  that applies, gates, and versions through the existing on-device surface unchanged.
- **`FederatedImprovement` + app surface.** `app.contribute_federated(member_id=,
  participants=, training_set=|runs=)` builds this member's contribution behind the
  consent ledger's TRAINING purpose and the residency posture;
  `app.adopt_federated(dataset, contributions, training_set=|runs=)` runs the gated
  round end to end — securely aggregate → refit the member's own adapter → gate it
  against the base on the held-out set (at-least-as-good, the same no-regression and
  canary discipline a local promotion clears) → adopt + apply or refuse + roll back
  (returning a `FederatedRoundResult`). `app.federated_improvement(...)` returns the
  streaming controller (`observe → aggregate → refit → gate → adopt / rollback`).
  Every decision lands on the hash-chained audit log and the event bus.
- **`federated` VincioBench family + SLOs.** Measures the no-raw-traffic guarantee,
  bounded sensitivity, secure-aggregation mask cancellation and individual hiding,
  k-anonymity refusal, deterministic federated PCA, fleet coverage, the
  at-least-as-good no-regression gate, live grounded answering, reversibility, and
  refusal of a regressing federated adapter. Two new published privacy SLOs and a
  no-regression SLO gate it. Runnable example
  `59_federated_cross_org_self_improvement.py`.

## [3.14.0] - 2026-06-22

On-device fine-tuning & continual local adaptation. The distillation flywheel
already turns production traces into executed *hosted* fine-tune jobs, and the
in-process GGUF provider already runs a quantized model air-gapped; this release
adds the rung between them — **local adaptation**, a LoRA-class adapter fit
*on-device* from the same grounded data and applied to the in-process model, so an
air-gapped or edge deployment improves on its own traffic with no hosted training
round-trip and no traffic leaving the process. Entirely additive and
backward-compatible — `API_VERSION` stays `3.0`, the dependency-free offline path
is the default, and nothing below runs unless you opt in.

### Added

- **`LocalAdapter` (`vincio.optimize.local_adaptation`).** A versioned,
  content-addressed, portable LoRA-class adapter — the on-device analogue of a
  `.safetensors` LoRA file. Low-rank by construction (an `r×d` orthonormal basis
  plus an `n×r` code matrix and grounded targets) and bounded by an acceptance
  `gate` and a `scale` alpha (`scale=0.0` neutralizes it without unloading).
  `adapter.apply(query_vector)` is the forward pass: it scores a request against
  the learned subspace and returns the grounded answer only when the match clears
  the gate, deferring to the base model otherwise. `adapter.digest` is the
  behaviour-tracking content address; `save()` / `load()` write a portable JSON
  artifact.
- **`LocalLoRATrainer`.** Fits a `LocalAdapter` on-device from a grounded
  `TrainingSet` — `await trainer.fit(training_set, base_model)` embeds each
  example's prompt, builds a deterministic rank-`r` orthonormal subspace, and
  stores the projected codes alongside the grounded targets. Pure-Python and
  dependency-free; inject a `NativeLoRABackend` to additionally produce a real
  quantized GGUF/LoRA file on-device (loaded via the new `GGUFProvider(lora_path=,
  lora_scale=)`).
- **`AdaptedProvider`.** Wraps any `ModelProvider` (the in-process GGUF model, the
  deterministic mock, a hosted endpoint) so an in-distribution request is answered
  the grounded way the adapter learned and everything else falls through to the
  base model unchanged. Transparent: it reports the base provider's name and
  capabilities, so residency, provenance, and the rotation stack are unaffected.
- **`AdapterRegistry`.** A versioned, reversible store of on-device adapters —
  `register` assigns the next version and makes it the active head (storing an
  independent copy), `rollback` restores an earlier version, and an optional
  on-disk directory persists every version and the head pointer across restarts.
- **`AdapterGate`.** The no-regression gate for an on-device adapter — the
  model-swap gate's analogue — reusing the same `CanaryVerdict` machinery a prompt
  deploy and a model rotation clear: an adapter promotes only when the adapted
  model is at-least-as-good as its base on a held-out set, with no significant
  regression.
- **`ContinualAdaptation` + app surface.** `app.adapt_locally(dataset, runs=|
  training_set=, policy=LocalAdaptationPolicy())` runs the gated loop end to end —
  curate the grounded data, fit an adapter on-device, gate it against the base, and
  on a pass register + apply it (returning an `AdaptationResult`); a regressing
  adapter is refused and the registry head left on the last known-good version.
  `app.local_adaptation(...)` returns the streaming `ContinualAdaptation`
  controller (`observe → train → gate → promote / rollback`), and
  `app.use_local_adapter(adapter)` / `app.use_local_adapter(None)` apply or unload
  one live. Every decision lands on the hash-chained audit log and the event bus.
- **`local_adaptation` VincioBench family + SLOs.** Measures on-device low-rank
  fitting, bounded in-/off-distribution application, the at-least-as-good
  no-regression gate, live grounded answering, reversibility, refusal of a
  regressing adapter, deterministic content-addressing, and versioned rollback.
  Three new published SLOs gate it. Runnable example
  `58_on_device_local_adaptation.py`.

## [3.13.0] - 2026-06-22

Learned semantic cache & near-miss KV reuse. Exact-match prompt caching already
serves a byte-identical request for free; this release adds the rung above it —
**near-miss reuse**, answering a request that is *semantically equivalent* (not
byte-identical) to a recent one straight from cache, with the acceptance threshold
*learned from the platform's own traces* so a near-miss is served only when it is
safe — never below the bar. Entirely additive and backward-compatible —
`API_VERSION` stays `3.0`, the dependency-free offline path is the default, and
nothing below runs unless you opt in.

### Added

- **`LearnedSemanticCache` (`vincio.caching`).** A bounded, calibrated, auditable
  near-miss response cache. `await cache.lookup(query, policy_scope=, schema_ref=)`
  embeds the query, scans the entries that share its scope (model + stable prompt
  head) and output schema, and serves the most-similar unexpired entry **only when
  its similarity clears the calibrated acceptance threshold** — a below-bar best
  match is a recorded-but-never-served near-miss. `await cache.store(query, value,
  policy_scope=, schema_ref=, response_tokens=)` populates it. Bounded LRU under the
  resident-memory budget (`SemanticCachePolicy.max_entries` / `max_resident_bytes`),
  surfaced by `cache.stats()` (a `SemanticCacheStats`). Deterministic insertion
  order; freshness reads an injectable clock.
- **Trace-calibrated acceptance threshold.** `ThresholdCalibrator` /
  `cache.calibrate(examples)` / `await cache.calibrate_from_pairs([(q, q2,
  equivalent), ...])` fit the **lowest** threshold (at or above `min_floor`) whose
  accepted set clears `target_precision`, returning a `CalibrationReport`. When the
  target is unreachable the threshold falls back to `1.0` (near-miss serving
  effectively off) with `calibrated=False`, rather than guess — the "never serve
  below the bar" guarantee.
- **Auditability & reversibility.** Every accepted near-miss is recorded as a
  `SemanticCacheHit` (`cache.audit()`), and any entry can be rolled back with
  `cache.revoke(key)`; `cache.clear()` participates in the `InvalidationManager` so
  a policy / schema / scope change clears the cache like the exact-match caches.
- **`SemanticCacheGate`.** The cache analogue of the model-swap `SwapGate`:
  `await gate.evaluate(cache, [SemanticGateCase(...)])` replays probe cases through
  the cache and checks every served near-miss is at-least-as-good as the live answer
  at a fixed budget (a pluggable scorer; dependency-free `lexical_quality` default),
  so a drifted cache is caught before it ships.
- **`KVPrefixPool` (`vincio.caching`).** Cross-request reuse of a shared
  stable-prefix KV footprint: `pool.observe(prefix_hash=, model=, prefix_tokens=)`
  reports whether a request reused a warm head and the serving-engine KV the shared
  head avoids recomputing (`pool.report()` → `KVReuseReport`), bounded LRU under the
  resident budget.
- **App integration.** `app.use_semantic_cache(policy_or_cache=None)` and
  `app.use_kv_prefix_reuse(pool=None)` install both layers (also enabled from
  `cache.semantic_cache` / `cache.kv_prefix_reuse` config); the runtime consults
  them on the live path only when installed (model spans gain `semantic_cached` /
  `kv_prefix_reused` / `kv_bytes_reused`). Reports via `app.semantic_cache_report()`
  / `app.kv_prefix_report()`; a served near-miss is a $0-billed call in the cost
  report.
- **`semantic_cache` VincioBench family + SLOs.** Measures trace calibration,
  near-miss serving above the bar, below-bar refusal, at-least-as-good hit quality
  served through the run path, the eval-replay gate blocking a drifted cache,
  cross-request KV reuse, and resident-budget bounding. Three new published SLOs
  gate it. Runnable example `57_learned_semantic_cache.py`.

## [3.12.0] - 2026-06-21

Causal record-replay debugger. The eval-replay runner and durable-graph
time-travel already let a run be re-executed from a checkpoint or a recorded
case; this release adds the rung the platform had not yet made first-class —
**byte-faithful, deterministic replay of a *whole* agent run from its trace**, so
a past run becomes something you can step, inspect, and branch instead of a
bespoke script. Entirely additive and backward-compatible — `API_VERSION` stays
`3.0`, the dependency-free offline path is the default, and nothing below runs
unless you opt in.

### Added

- **`Recorder` (`vincio.observability`).** `Recorder(app).record(input)` runs an
  app while capturing every non-deterministic edge of the run — model responses
  (keyed by `ModelRequest.hash`), tool outputs (by name + canonical arguments),
  retrieval hits (by query + params), the `ModelCapabilities` each request was
  negotiated against, and the clock/seed — into a portable `Recording`. The run
  executes normally against the real provider/tools/retrieval; capture is done by
  shadowing `resolve_provider`, `tool_runtime.execute`, and `retrieval.retrieve`
  for the run and restoring them after.
- **`Recording` (`vincio.observability`).** A self-contained, JSON-serializable
  artifact carrying the recorded edges, the full trace span tree, and a
  `fidelity_digest`. It is content-addressed and verifiable —
  `recording.put(store)` / `Recording.from_store(store, address)` write to / load
  from any `EvidenceStore`, `recording.save(path)` / `Recording.load(path)` use a
  file, and `recording.verify()` recomputes the digest and every edge's content
  address so a tampered or truncated recording is caught before replay. Rich
  inspection surface: `model_calls` / `tool_calls` / `retrievals`, `steps()` over
  the span tree, and `render_text()`.
- **`Replayer` (`vincio.observability`).** `Replayer(app).replay(recording)`
  re-executes a recording against an app, serving every edge from the recording
  so the run reproduces **byte-for-byte** — the recording, not the live provider,
  drives the run. The `ReplayResult` is `faithful` only when no edge diverged and
  the output is byte-identical, and lists every `Divergence` (the edge live code
  asked for that was not in the recording) — so changed code is detected and
  reported, never silently re-executed. The underlying `ReplayProvider` serves
  recorded model responses by request identity.
- **Branch-and-edit.** `Replayer(app).branch(recording, edits=[BranchEdit(...)],
  input=, fallback=)` forks a recording, changes a recorded edge or the input,
  and re-executes **only the affected suffix** while the unchanged prefix is
  still served from the recording (`served_from_recording` vs `reexecuted`), so a
  fix is validated against the exact failing run.
- **`record_replay` VincioBench family + SLOs.** Measures byte-identical replay
  against a live provider that would answer differently, divergence detection
  when the prompt changes, the content-addressed store round-trip and fidelity
  verification, and branch-and-edit prefix-reuse / suffix-re-execution. Three new
  published SLOs gate it. CLI: `vincio trace verify-recording <file>`. Runnable
  example `56_record_replay_debugger.py`.

## [3.11.0] - 2026-06-21

World-model / simulation-based planning. The stateful-environment harness and the
test-time-search verifiers already let an agent *evaluate* a trajectory against the
live world; this release adds the rung above it — letting an agent **learn a model
of its tools and plan against it**, searching imagined rollouts before acting so a
wrong move costs a simulated step, not a live one. Entirely additive and
backward-compatible — `API_VERSION` stays `3.0`, the dependency-free offline path is
the default, and nothing below runs unless you opt in.

### Added

- **`WorldModel` (`vincio.agents`).** A deterministic, offline dynamics model fit
  from recorded reset/step `Transition`s (`record_transitions(env, sequences)`). For
  each tool it learns the *parameterized* state effect — whether a changed value is
  a constant, an argument, or a numeric step — under a *learned precondition* (the
  discriminative state field that decides which effect fires), so it predicts a
  refund will *fail* on a processing order and *succeed* on a cancelled one, and
  generalizes a cancel it only ever saw on one order to another. `predict(obs,
  action)` returns a `PredictedStep` (predicted next observation, reward, ok,
  confidence); an unseen action signature predicts the identity with zero
  confidence. `imagine(obs, actions)` rolls a whole plan forward without touching a
  tool.
- **`CalibrationReport` (`WorldModel.calibrate`).** The world model earns planning
  weight only after its predicted next states and rewards track the real environment
  within a tolerance, the way a judge ensemble earns gating weight — reporting
  next-state accuracy, reward MAE, and a `trusted` verdict the planner checks.
- **`ModelPredictivePlanner` (`vincio.agents`).** A receding-horizon (MPC) planner
  that searches imagined rollouts under the world model with the test-time-search
  beam, commits the best **first** action to the real environment, observes, and
  re-plans — so model error is corrected every step. The beam score prefers the
  shortest, cheapest plan that reaches the goal (cost-aware action selection); by
  default it refuses an uncalibrated model. Returns an `MPCResult` (`MPCStep`s, the
  committed actions, the oracle verification, the earned planning weight).
- **`make_vault_environment` (`vincio.evals.environment`).** A planning-favoring
  reference world: a locally-attractive `shortcut` raises the task score immediately
  but seals the vault shut, so only a planner that rolls the model forward avoids
  the dead end. The `task_goal_value` helper scores an observation by the fraction
  of a task's checks it satisfies.
- **`world_model` VincioBench family + SLOs.** Measures the learned dynamics
  (next-state accuracy, learned precondition, argument generalization), the
  calibration gate, and the planning-accuracy guarantee: on the vault world the
  imagined-rollout planner opens the vault while a reactive (one-step) planner is
  trapped at a fixed action budget. Three new published SLOs gate it. Runnable
  example `55_world_model_planning.py`.

## [3.10.0] - 2026-06-21

Long-horizon context engineering. Vincio's namesake is context engineering, and
the regime where it matters most is the one naïve accumulation breaks:
million-token, multi-day, multi-session agent runs where stale context crowds out
fresh signal ("context rot") and the resident footprint grows without bound. This
release composes the platform's existing primitives — the footprint estimator, the
memory decay model, and the content-addressed evidence store's cross-process
`materialize()` — into an explicit per-run context governor. Entirely additive and
backward-compatible — `API_VERSION` stays `3.0`, the dependency-free offline path
is the default, and a run with no governor installed behaves exactly as before.

### Added

- **`ContextGovernor` (`vincio.context`).** A per-run controller that holds a
  `ContextBudget` (live tokens, resident bytes, KV-cache footprint) across a long
  run the way the cost report holds a dollar budget. On each admission it
  re-applies intra-run decay, then — while over budget — compacts the coldest
  non-recent spans into the memory OS (or evicts the lowest-utility span when no
  compactor is configured) until the live footprint fits. `recall(query)` answers
  over the live spans and **pages cold detail back** from the summaries that cover
  it, so recall survives compaction. `report()` returns a `ContextBudgetReport` —
  the residency analogue of the cost report.
- **`RelevanceDecay` (`vincio.context`).** The memory subsystem's exponential
  decay model applied *within a single run*: a span admitted many steps ago keeps
  `0.5 ** (age / half_life_steps)` of its base relevance, so fresh signal outweighs
  stale signal of equal base relevance. Demotions are surfaced in the
  excluded-context report.
- **`ContextCompactor` (`vincio.context`).** Hierarchical, provenance-preserving
  compaction: folds a batch of cold spans into one extractive summary span whose
  full source text is written to a content-addressed `EvidenceStore` (paged back
  losslessly on demand) and whose gist is written into the memory OS as an audited
  `SUMMARY` memory carrying the covered content hashes and source ids. Because a
  summary is itself a span, summaries compact again into higher levels. A
  `CompactionRecord` captures the provenance of each fold.
- **App wiring.** `app.use_context_governor(budget_or_governor, ...)` installs a
  governor (its compactor writes summaries into the app's memory engine);
  `app.govern_packet(result_or_packet)` admits a run's evidence; and
  `app.context_budget_report()` returns the live footprint.
- **`long_horizon` VincioBench family + SLOs.** Measures the horizon-scaling
  guarantee: at 10× horizon the governed resident/token footprint stays flat (vs
  the ~linear growth of naïve accumulation), a compacted needle is still recalled
  by paging it back, provenance is retained through compaction, and intra-run decay
  demotes stale spans. Three new published SLOs gate it. Runnable example
  `54_long_horizon_context.py`.

### Changed

- The agent executor's internal in-loop compactor (`vincio.agents.compaction`) is
  renamed `ContextCompactor` → **`LoopCompactor`** to reserve the `ContextCompactor`
  name for the new long-horizon context-layer class. It was never part of the
  public surface (not in `vincio.__all__`); the executor and benchmarks are updated
  in lockstep. `vincio.context.longhorizon` joins the `mypy --strict` ladder.

## [3.9.0] - 2026-06-21

Test-time compute & reasoning orchestration. Reasoning-model thinking budgets and
parallel test-time search are the cheapest quality lever left, and the platform
already owned the pieces to orchestrate them — cost-aware action selection,
critics and judge ensembles that act as verifiers, and a provider-neutral
reasoning-effort knob. This release makes test-time compute a first-class,
budgeted, cache-aware dimension of the run. Entirely additive and
backward-compatible — `API_VERSION` stays `3.0`, the dependency-free offline path
is the default, and every existing entry point is unchanged (a run with no
reasoning controller installed and no `reasoning_effort` pinned behaves exactly as
before).

### Added

- **`ReasoningController` (`vincio.agents`).** A deterministic policy that sets
  the thinking effort and a thinking-token budget per step from the task
  classification and the live budget (reusing the same difficulty estimator that
  drives the capability-aware router). A `ReasoningPolicy` configures the
  difficulty→effort bands and the guardrails: a **hard `max_reasoning_tokens`
  ceiling** and a `budget_fraction` cap on a share of the remaining output budget,
  so a hard task can never silently exhaust the run. Low prior confidence
  escalates one level; a warm thinking prefix steps it down. `decide(...)` returns
  an explainable `ReasoningDecision`. `app.use_reasoning_controller(...)` installs
  one so the runtime fills an unset `reasoning_effort` per run and records the
  choice on the trace (`reasoning_source`, `reasoning_reason`); `app.reasoning()`
  builds one to call directly.
- **Reasoning-trace-aware caching (`vincio.caching`).** `ReasoningTraceCache` is a
  byte-budgeted LRU of paid thinking prefixes keyed by stable-prefix hash + model
  + effort (`reasoning_prefix_key`), evicting LRU-first under both an entry count
  and a resident-byte ceiling. The runtime records each paid reasoning trace, so a
  re-ask that shares a thinking prefix is recognized as warm and its effort stepped
  down — the reasoning analogue of the compiled-prompt render program.
- **`TestTimeSearch` & the `Verifier` protocol (`vincio.optimize`).**
  Verifier-guided **best-of-N**, **self-consistency**, and **beam search** over
  tool-use trajectories. Candidates are scored by the platform's *existing*
  critics through one `Verifier` protocol: `JudgeVerifier` wraps any `Judge` /
  `JudgeEnsemble` (a split panel's disagreement lowers its confidence),
  `RewardVerifier` wraps any `VerifiableReward` / `RewardModel`, and
  `CallableVerifier` wraps a plain function. Best-of-N early-exits the moment the
  verifier clears the bar; self-consistency early-exits the moment the majority is
  mathematically locked. Bounded by a `SearchBudget` (candidate / cost / deadline).
  `app.test_time_search(input, *, verifier=, strategy=, n=)` runs it over a varied
  re-run of the app.
- **`test_time_compute` VincioBench family + SLOs.** Measures the quality-per-dollar
  trade: a Pareto quality gain over single-shot at a fixed budget, quality per cent
  of spend, early-exit savings, self-consistency accuracy lift, and that the
  reasoning controller's hard token ceiling holds across every difficulty. Three
  new published SLOs gate it. Runnable example `53_test_time_compute.py`.

## [3.8.0] - 2026-06-21

Provable prompt-injection containment & capability-secure agents. The security
subsystem already *detects* injection, RAG-poisoning, secrets, and PII; this
release adds the containment that holds even when detection misses, by separating
the control plane from the data plane in the library's own provenance and
permission model. Entirely additive and backward-compatible — `API_VERSION` stays
`3.0`, the dependency-free offline path is the default, and every existing entry
point is unchanged.

### Added

- **Information-flow labels & taint propagation.** `TrustLabel` promotes
  provenance to a typed `trusted` / `untrusted` / `quarantined` lattice (`join`
  takes the least-trusted, so taint never decreases). `TaintedValue` carries a
  value with its label and provenance sources and propagates the label through
  `map` / `derive`, so a value computed from any untrusted input is itself tainted
  and cannot be laundered back to trusted. `TrustLabel.from_trust_level` bridges
  the existing `TrustLevel` provenance.
- **Unforgeable capability tokens.** `CapabilityToken` is an HMAC-signed,
  principal- and argument-scoped, TTL-bounded grant minted by a `CapabilityBroker`
  from the *user's* request — never from model output. `CapabilityBroker.verify`
  (constant-time signature compare) returns an explainable `CapabilityVerification`;
  a token minted under a different secret, tampered with, expired, or used outside
  its pinned argument constraints never verifies.
- **Dual-plane execution.** `DualPlaneExecutor` wraps the permissioned
  `ToolRuntime`: untrusted bytes are held in a quarantine (`QuarantineRef`), the
  privileged planner sees only typed, schema-validated `extract`ions (and
  `control_messages` never contain the bytes), and every side-effecting `call`
  whose arguments carry an untrusted taint is refused unless it presents a valid
  capability or an approval. Tool output is re-quarantined so taint propagates
  across steps. New `ContainmentError` (`CONTAINMENT_BLOCKED`) with a catalog
  entry.
- **Machine-checkable containment invariant.** `ContainmentMonitor` records each
  capability exercise as a `ContainmentEvent`; `verify_containment` folds the log
  into a `ContainmentReport` whose `held` is true iff
  `untrusted ⇒ no unapproved capability` held for every decision, with an
  `escalation_rate` over untrusted side-effecting attempts.
- **Capability-scoped tools at the permission layer.** `ToolPermissionChecker`
  gains an opt-in `broker=` / `require_capability=`; with a broker configured a
  side-effecting tool whose arguments are untrusted-tainted (or whose taint is
  unknown) must present a capability, else it is routed to the approval gate.
  `ToolRuntime.execute(..., capability=)` threads the token through. Without a
  broker the prior RBAC/ABAC behavior is unchanged.
- **Taint-propagating materialization.** `ContextPacket` carries each evidence
  entry's `trust_level`; `materialize()` stamps a derived `trust_label`, and
  `tainted_evidence()` returns each evidence text as a labeled `TaintedValue`.
- **VincioBench & SLO.** New `containment` family runs an adversarial
  injection corpus through the dual-plane executor; a published SLO holds the
  escalation rate at **0** on the gated corpus, backed by budgets that also gate
  taint propagation, planner isolation, capability unforgeability, and that
  legitimate capability-authorized side effects still run.
- **Runnable example.** `examples/52_injection_containment.py` walks
  information-flow labels → quarantine + typed extraction → capability-gated
  execution → the machine-checked containment invariant, fully offline.

## [3.7.0] - 2026-06-20

The learning loop, closed with on-policy reinforcement. Reinforcement from
verifiable rewards (RLVR) turns the signals the platform already computes into a
reward that improves a *policy*, not just a prompt — without adding a trainer
dependency to the default path. Entirely additive and backward-compatible —
`API_VERSION` stays `3.0`, the dependency-free offline path is the default, and
every existing entry point is unchanged.

### Added

- **Verifiable reward model.** `RewardModel` composes one or more
  `VerifiableReward`s into a dense, confidence-weighted signal. `OracleReward`
  reads the stateful-environment task-success oracle (dense partial credit or
  pass/fail); `BenchmarkReward` turns any of the nine `BenchmarkAdapter` scorers
  (exact / contains / `pass@1` / solvable-path) into a reward; and
  `JudgeEnsembleReward` turns a judge panel into a reward whose **disagreement
  down-weights itself** (`weight = 1 − spread`), so a split panel leans the blend
  on the verifiable scorers rather than rewarding noise. `RewardSignal` /
  `RewardSample` carry the value, the verifiable `success`, the confidence weight,
  and provenance. New `RewardError` with a catalog entry.
- **Step-level credit assignment.** `TrajectoryAdvantage` attributes a
  trajectory's outcome reward back to the steps that earned it by Shapley
  counterfactual replay — `environment_step_value` re-verifies the environment end
  state with only the kept tool steps, so a step the success depended on earns its
  marginal. Credits sum to the attributable value (efficiency) and `StepCredit`
  reports each step's signed contribution and share.
- **The trajectory optimizer (`app.learn`).** `TrajectoryOptimizer` runs a
  GRPO-style group-relative update (`compute_group_advantages`) over a
  deterministic `SoftmaxPolicy`, behind the same safety discipline prompt
  optimization uses: a **KL-to-reference clamp** (`kl_divergence`, a binary-search
  projection back into the trust region) and a **monotonic no-regression gate**
  (`no_regression_gate`) so the served policy never regresses the baseline reward.
  `app.learn(tasks, reward=, ...)` returns a `LearningResult` whose `verdict` is
  the same `CanaryVerdict` a prompt deploy produces; the decision is audited
  (`learn.promoted` / `learn.rejected`). On a promotion the on-policy winners are
  exported as a grounded `TrainingSet`, and a configured `flywheel` emits a
  fine-tune job through the existing distillation flywheel in the same call.
- **Shared Shapley kernel.** `vincio.core.shapley` (`shapley_values`,
  `ashapley_values`, `shapley_from_cache`, `is_efficient`, `coalitions`) is the
  pure, dependency-free credit-assignment kernel now shared by both
  `TrajectoryAdvantage` and the causal regression `CausalAttributor`.
- **Runnable example.** `examples/51_reinforcement_from_verifiable_rewards.py`
  walks verifiable rewards → step-level credit → the gated optimizer → emitting a
  fine-tune job, fully offline.

### Changed

- `RewardModel`, `VerifiableReward`, `TrajectoryAdvantage`, `TrajectoryOptimizer`,
  and `LearningResult` are re-exported from the top-level `vincio` namespace; the
  full reward / advantage / policy / optimizer surface is exported from
  `vincio.optimize`.
- `CausalAttributor` now computes its Shapley decomposition through the shared
  `vincio.core.shapley` kernel instead of an inlined loop (behavior unchanged).
- VincioBench gains a `learning` family with three SLOs — reward-monotonicity
  (`rlvr_reward_monotonicity`), KL-bound adherence (`rlvr_kl_bound_adherence`), and
  no-regression-vs-baseline (`rlvr_no_regression_vs_baseline`) — backed by budgets
  that also gate the Shapley step-credit efficiency, the judge-disagreement
  down-weighting, and the on-policy flywheel emission offline.

## [3.6.0] - 2026-06-20

Evaluation & quality frontier: measure more of what buyers compare on, and
explain regressions instead of just flagging them. Entirely additive and
backward-compatible — `API_VERSION` stays `3.0`, the dependency-free offline path
is the default, and every existing entry point is unchanged.

### Added

- **Four more benchmark adapters.** `AgentBenchAdapter`, `ToolBenchAdapter`,
  `LiveCodeBenchAdapter`, and `MMLUProAdapter` join the existing five behind the
  same `BenchmarkAdapter` contract — nine in all — each scored by the benchmark's
  own verifiable scorer: AgentBench's per-environment exact / contains / set /
  numeric match, ToolBench's solvable pass-rate over a call path (terminate with an
  answer, no hallucinated APIs, gold-answer match), LiveCodeBench's all-tests-pass
  (`pass@1`) over recorded per-test outcomes, and MMLU-Pro's A–J option-letter
  extraction-and-match. Each is task-set-hash pinned, replayable from a shipped
  fixture, and loadable from the official export format via
  `agentbench_tasks_from_export` / `toolbench_tasks_from_export` /
  `livecodebench_tasks_from_export` / `mmlu_pro_tasks_from_export`.
- **Judge ensembles with disagreement detection.** `JudgeEnsemble` scores a panel
  of judges together (`"mean"` / `"median"` / outlier-robust `"trimmed_mean"`),
  surfaces their spread as an uncertainty signal (`judge_disagreement`,
  `EnsembleVerdict.uncertain` / `.spread`), and is calibrated as a whole against
  human labels — `calibrate()` records the panel-vs-human Cohen's κ that
  `gating_weight()` gates on, so a split or uncalibrated panel cannot block CI.
- **Causal regression attribution.** `CausalAttributor` / `attribute_regression`
  attribute a metric delta to the components a release changed (prompt / retrieval
  / model / budget, declared as `AttributionFactor`s) by **Shapley counterfactual
  replay** over all `2**k` baseline/candidate coalitions. The resulting
  `AttributionReport` names the dominant cause and how concentrated the blame is;
  its per-factor `FactorContribution`s sum exactly to the total delta (efficiency),
  splitting interactions fairly rather than double-counting them.
- **Adaptive eval sampling.** `AdaptiveSampler` decides a mean-aggregate gate with
  the fewest samples by seeding every case, then allocating each next sample to the
  highest-variance case (Neyman-optimal, run sequentially) and stopping the moment
  the confidence interval clears the threshold. `AdaptiveSamplingResult` reports the
  verdict, the CI, the per-case allocation, and the savings — the same verdict as
  the exhaustive run, for fewer samples.

### Changed

- `JudgeEnsemble`, `CausalAttributor`, `attribute_regression`, and
  `AdaptiveSampler` are re-exported from the top-level `vincio` namespace; the full
  set (adapters, ensemble, attribution, adaptive types) is exported from
  `vincio.evals`.
- VincioBench's `agentic_evals` family gains a `quality_frontier` block and folds
  the four new adapters into its determinism check (now nine); new budgets and
  three SLOs (`judge_ensemble_calibration_gated`, `causal_regression_attribution`,
  `adaptive_sampling_preserves_verdict`) gate the guarantees offline.

## [3.5.0] - 2026-06-20

Professionalism & API ergonomics: make the platform's public surface as
trustworthy as its internals. Entirely additive and backward-compatible —
`API_VERSION` stays `3.0`, the dependency-free offline path is the default, and
every existing entry point is unchanged.

### Added

- **Actionable, internationalizable errors.** Every `VincioError` now carries a
  `.remediation` hint and a `.docs_url` deep link alongside its stable `.code`,
  resolved from a new completeness-gated catalog (`vincio.core.error_catalog`:
  `ERROR_CATALOG`, `catalog_entry`, `title_for`, `remediation_for`,
  `docs_url_for`, `render_error_reference`). An i18n layer
  (`register_error_locale` / `set_default_error_locale` / `available_error_locales`,
  English shipped as the reference locale) keys translated titles and hints by the
  same codes. `VincioError.to_dict()` includes `remediation` and `docs_url`, and a
  per-instance `hint=` / `docs_url=` override is available. `BenchmarkError` and
  `SkillError` gain their own codes (`BENCHMARK_ERROR` / `SKILL_ERROR`). New
  reference page `docs/reference/errors.md`, generated from the catalog and gated.
- **Versioned config migrations.** `VincioConfig` gains `schema_version`;
  `vincio.core.config_migrations` (`CONFIG_SCHEMA_VERSION`, `Migration`, `migrate`,
  `needs_migration`, `detect_version`) chains ordered, idempotent transforms.
  `load_config` upgrades stale files **in memory** so a config never silently
  drifts; `vincio config migrate [path] [--check] [--dry-run] [--output]` persists
  the upgrade, reporting each step and preserving the editor schema hint. The v0→v1
  migration introduces versioning and canonicalizes the legacy
  `observability.exporter: console` alias.
- **`vincio doctor`.** A static project scanner (`vincio.cli.doctor`:
  `run_doctor`, `collect_deprecations`, `scan_source`, `scan_config`) reports a
  project's use of any deprecated public API — its replacement and removal version
  read from the same `stability_of` metadata the library marks its own surface with
  — plus a `vincio.yaml` behind the current schema. AST-based: it never imports or
  runs project code.
- **Docstring-driven API reference + coverage gate.** `vincio._apiref`
  (`public_symbols`, `undocumented_symbols`, `render_api_index`) generates the
  exhaustive `docs/reference/api-generated.md` from `vincio.__all__`; a gate keeps
  every public symbol documented (`ContextApp`, `Crew`, `MemoryEngine`,
  `OutputSchema`, `Workflow` gained docstrings).
- **Strict typing.** The package ships a PEP 561 `py.typed` marker, so downstream
  type-checkers see Vincio's inline contract. A graduated, CI-enforced
  `mypy --strict` ladder covers `stability`, `core.errors`, `core.error_catalog`,
  `core.config`, `core.config_migrations`, `_apiref`, and `cli.doctor`, enforced by
  per-module overrides plus a dedicated CI step. New reference page
  `docs/reference/typing.md`.

### Changed

- The docs-completeness gate now also enforces docstring coverage and
  error-catalog completeness. A new `professionalism` VincioBench family gates
  these invariants under budgets. New runnable example `49_professionalism.py`.

## [3.4.1] - 2026-06-20

### Fixed

- **Fail-closed vertical-pack residency.** Resolve the active provider's real
  name when evaluating a residency posture so a vertical pack refuses an
  identifiable out-of-jurisdiction endpoint instead of passing on a name mismatch.

## [3.4.0] - 2026-06-20

Use-case coverage & verticals: go from primitives to a working app in one file,
in more domains. Entirely additive and backward-compatible — `API_VERSION` stays
`3.0`, the dependency-free offline path is the default, and every existing entry
point is unchanged. The four existing domain packs and their behavior are
untouched; the new capabilities sit behind new entry points or new pack names.

### Added

- **Vertical packs.** Five full-stack packs — `healthcare` (PHI), `ediscovery`
  (legal e-discovery), `kyc` (financial KYC/AML), `customer_support`, and
  `code_review` — preconfigure retrieval knobs, scoped memory, deterministic
  rails, domain metrics, an in-jurisdiction data-residency posture, and a golden
  eval set in one `app.use_pack(...)`. The `Pack` contract gains additive fields
  `retrieval` / `memory` / `residency` / `purpose` (wired through the public app
  API on `apply`) plus `Pack.is_vertical` and `Pack.retrieval_mode()`. A
  residency-pinned pack applies `set_residency([...region, "on_prem"],
  deny_on_unknown=False)` so the offline path runs while an identifiable
  out-of-jurisdiction endpoint is still refused. Each ships a golden eval set via
  `pack.dataset()` and a runnable example.
- **Assistant.** `app.assistant(...)` returns a conversational, session-aware
  `Assistant` over `ContextApp`: every `send` / `asend` is a full `run` threaded
  under one `session_id`, with multi-turn state carried by session-scoped memory
  write-back and a tool-approval surface (write tools denied and surfaced as
  `pending_approvals` until `approve(...)`, an `auto_approve` allow-list, or an
  `on_approval` callback grant them). Returns an `AssistantTurn`
  (`text` / `output` / `citations` / `approvals` / `memory_writes` / `trace_id` /
  `cost_usd`); `history()` / `reset()`; satisfies the `Simulator` agent contract
  for multi-turn evaluation. New public surface: `Assistant`, `AssistantTurn`,
  `ApprovalRecord` (`vincio.assistant`).
- **End-to-end voice agent.** `app.voice_agent(...)` returns a `VoiceAgent`
  (`vincio.realtime`) that wires a realtime session to the deep-research agent (an
  in-session, cited `research` tool), the self-editing memory OS, and the app's
  deterministic input/output rails over every spoken transcript and reply. Tool
  calls route through the permissioned, sandboxed, audited runtime; the
  dependency-free in-process backend keeps it offline-testable.
- **Cookbook.** Task-shaped recipes ship as runnable, offline-gated examples:
  contract redlining (`45`), incident triage (`46`), data-room Q&A (`47`), and
  multimodal RAG over slides/PDFs (`48`), alongside capability examples for the
  vertical packs (`42`), the Assistant (`43`), and the voice agent (`44`).

### Changed

- **Structured-output redaction.** An output `redact` rail now masks detected
  PII/secrets in the string fields of a **structured** output (not only text
  outputs), preserving the schema and field types. This closes a gap where a typed
  deliverable could carry an identifier the rail had detected; the raw model
  emission on the trace is unchanged (trace content capture remains off by
  default).

## [3.4.1] - 2026-06-20

A correctness follow-up to 3.4.0's vertical packs: a regulated domain should fail
*closed* on an unresolvable region, not fail open. Fixing the root cause — the
provider name of a passed instance — makes that posture compatible with the
offline-first default. Backward-compatible; `API_VERSION` stays `3.0`.

### Fixed

- **Provider name from a passed instance.** When a `ModelProvider` *instance* was
  passed to `ContextApp(provider=...)`, the app recorded the provider name as the
  config default (`"openai"`) instead of the instance's real name. The name is now
  read from the instance (`provider.name`), so data-residency checks, C2PA
  provenance marking, and provider lookups reflect the actual provider — e.g. the
  deterministic mock and the local provider correctly resolve to the `on_prem`
  region. The string and default constructor paths are unchanged.

### Changed

- **Vertical-pack residency is now fail-closed.** Residency-pinned vertical packs
  (`healthcare`, `ediscovery`, `kyc`) apply `set_residency([...region, "on_prem"])`
  with `deny_on_unknown=True` (was `False`): a provider whose region cannot be
  resolved is refused egress — the correct posture for a regulated domain. The
  dependency-free offline path still runs because the mock / local providers now
  resolve to the known `on_prem` region (see the provider-name fix above); a live
  deployment makes its region known by pinning a region-bearing endpoint or
  declaring `provider_regions`.

## [3.3.0] - 2026-06-20

Ecosystem & integration breadth: meet teams where their data and tools already
live. Entirely additive and backward-compatible — `API_VERSION` stays `3.0`, the
dependency-free offline path is the default, and every existing entry point is
unchanged. New heavy integrations are opt-in extras; the new plugin contract is
versioned at `PLUGIN_API_VERSION = "1.0"`.

### Added

- **First-party connectors.** Eight new connectors feed the document engine with
  full provenance behind the existing `register_connector` / `connect` contract:
  `jira`, `linear`, `gdrive`, `sharepoint`, `salesforce`, and `zendesk` (REST,
  riding the core `httpx` dependency), plus `bigquery` and `snowflake` (warehouse,
  via an injected client/connection or the `vincio[bigquery]` / `vincio[snowflake]`
  extra). Each accepts an injected client so it round-trips offline; each returns
  `Document`s with `source_uri`, connector metadata, and timestamps. VincioBench
  gates `families.integrations.{connectors_round_trip,connector_provenance}`.
- **Entry-point plugin system.** A new `vincio.plugins` module formalizes a
  versioned plugin contract: third-party providers, embedders, stores, connectors,
  chunkers, rerankers, judges, metrics, and packs register themselves on install
  via the `vincio.<kind>` entry-point groups. `installed_plugins()` /
  `discover_plugins()` report without importing targets; `load_plugins()` registers
  compatible plugins (idempotent, isolating a broken one). A distribution may
  declare its targeted plugin-API major (`vincio.plugins:api_version`); a major
  mismatch is reported and skipped. `connect()` / `load_pack()` auto-load on a name
  miss; `vincio plugins list` (CLI). New `register_reranker` / `build_judge` /
  `register_judge` / `JUDGES` and `skill_from_markdown`. VincioBench gates
  `families.integrations.{plugin_loads_on_install,plugin_gates_incompatible}`.
- **Community pack & skill registry.** `vincio.registry.CommunityRegistry` is a
  signed, governed index of opt-in domain packs and `SKILL.md` skill bundles. Each
  `BundleRecord` is content-bound (SHA-256) and may be signed with the library's
  `ChainSigner` (HMAC, or Ed25519 for third-party verification); every resolution
  passes the same `AllowListGate` the agent fabric uses, verifies the signature,
  and is recorded as an audited `bundle_resolve` access decision — a tampered,
  unlisted, or unsigned-when-required bundle is denied, not served.
  `publish_pack` / `publish_skill` / `load_pack` / `load_skill` / `sign_index` /
  `verify_index`. VincioBench gates `families.integrations.{registry_resolution_
  governed,registry_resolution_audited,registry_signature_verified,registry_tamper_
  detected}`.
- **Deeper framework interop.** `vincio.interop` gains Haystack and DSPy bridges
  alongside LangChain / LlamaIndex: `from_haystack_document` /
  `from_haystack_retriever` / `from_haystack_embedder` / `add_haystack_component` /
  `to_haystack_document(s)`, and `from_dspy_module` / `from_dspy_retriever` /
  `from_dspy_signature` / `add_dspy_module` / `to_dspy_lm`. The `from_*` direction
  is duck-typed (no heavy import); `to_*` needs `vincio[haystack]` / `vincio[dspy]`.
  VincioBench gates `families.integrations.{haystack_bridge,dspy_bridge}`.
- **MCP-server marketplace bridge.** `app.add_mcp_from_registry(name, registry=,
  allow=/deny=/directory=)` composes discovery (`MCPRegistryClient`), governance (a
  governed `AgentDirectory` under an `AllowListGate`, recording an audited
  `agent_resolve` decision), and connection (the existing permissioned, sandboxed,
  audited runtime) in one call — a discovered server's tools land namespaced and
  enabled; an unlisted server raises `AccessDeniedError`. VincioBench gates
  `families.integrations.{mcp_marketplace_tool_landed,mcp_marketplace_audited,
  mcp_marketplace_denies_unlisted}`.

### Reliability

- New VincioBench `integrations` family: every connector and interop bridge
  round-trips offline against a recorded fixture (`benchmarks/fixtures/
  integrations.json`), the plugin contract is exercised end-to-end, and the
  registry resolution is verified to be an audited access decision. Example
  [`41_ecosystem_and_integration.py`](examples/41_ecosystem_and_integration.py).

**1761 tests passing offline; ruff + mypy clean; VincioBench 315 budgets / 101 SLOs.**

## [3.2.0] - 2026-06-20

Orchestrator & planner depth: make multi-step execution plan better, recover
from failure, and schedule fairly at scale. Entirely additive and
backward-compatible — `API_VERSION` stays `3.0`, the dependency-free offline path
is the default, and every existing planner / graph / agent path is unchanged.
The event-catalog schema bumps to `3.1` for the new `plan.repaired` payload.

### Added

- **Hierarchical (HTN) planning.** A new `hierarchical` planner mode decomposes a
  goal into a sub-goal tree and binds each leaf to a bounded step, composable with
  the existing planners. `vincio.agents.HTNDomain` (`.method(task, subtasks,
  ordering="sequence|parallel", when=)` / `.operator(name, step_type=, tool_name=,
  fallbacks=)`) decomposes deterministically into an `HTNPlanNode` tree that
  `dag_from_plan_node` flattens into an executable `StepDAG`; without a domain the
  model proposes a two-level decomposition, with a static fallback offline.
  `app.agent(planner="hierarchical", domain=...)`. VincioBench gates
  `families.agent.planner_depth.hierarchical_parallel`.
- **In-place plan repair.** On a tool failure, a validation contradiction, or a
  budget shock the executor edits the *remaining* plan instead of restarting —
  `vincio.agents.PlanRepairer` re-binds a failed tool to a `fallback_tools` /
  name-overlap alternative, substitutes a reasoning step when none exists,
  reorders a corrective re-analysis before the finalize, or drops the optional
  tail to finalize inside the budget. On by default (`AgentExecutor(repair=False)`
  to disable). Each repair is an `AgentState.repairs` entry, a typed
  `plan.repaired` event, and a `plan_repair` trajectory step. VincioBench gates
  `families.agent.planner_depth.repair_{rebind,substitute,budget_shock}`.
- **Cost-aware action selection.** `app.agent(cost_aware_models=[cheap, …,
  strong])` (or `vincio.agents.CostAwareSelector`) reads the data-driven
  `ModelRegistry` pricing and capabilities and the live budget to spend the
  cheapest capable model per step, escalating one tier only when the prior step's
  confidence is low; capability never traded for price. Each pick is a
  `SelectionDecision`. VincioBench gates
  `families.agent.planner_depth.cost_aware_savings` (≈ −57% vs always-strong).
- **Parallel sub-graph scheduling.** `vincio.agents.SubgraphScheduler` work-steals
  independent durable sub-graphs across the worker pool under one weighted
  fair-share budget (the shares sum to the cap), with a graph-level SLA deadline
  that returns the completed results plus the durable partial state of the rest
  rather than blowing the deadline. `.run([SubgraphTask(graph, input, weight=)])`
  → `ScheduleResult`. VincioBench gates `families.scale.subgraph.{speedup,
  fair_share_within_budget,deadline_returns_partial}`.
- **Durable timers & scheduled steps.** First-class `sleep_until` / `sleep_for` /
  `wait_for_event` node helpers pause a graph for a wall-clock delay, a webhook,
  or an approval without holding a worker; the wake condition rides the
  checkpoint, so it survives a restart. `TimerService(compiled).tick()` resumes
  due sleep timers and `.deliver(thread_id, event_name, payload=)` wakes an event
  wait (module-level `pending_timers` / `due_timers` / `resume_due_timers` /
  `deliver_event`). VincioBench gates
  `families.agent.planner_depth.durable_timer_restart_safe`.

### Reliability

- New SLOs: planner-repair recovery on a tool failure and a budget shock,
  cost-aware-selection savings (≥ 25%), parallel-sub-graph speedup (≥ 1.5×), and
  durable-timer restart safety — each backed by an at-least-as-strict VincioBench
  budget. Example [`40_orchestrator_planner_depth.py`](examples/40_orchestrator_planner_depth.py).

**1705 tests passing offline; ruff + mypy clean; VincioBench 301 budgets / 101 SLOs.**

## [3.1.0] - 2026-06-20

Runtime performance & efficiency: make the compile spine fast enough that
context engineering is never the bottleneck. Entirely additive and
backward-compatible — `API_VERSION` stays `3.0`, the dependency-free offline
path is the default, and every default run path is unchanged. NumPy is an
optional accelerator, never a requirement.

### Added

- **Vectorized candidate scoring.** `ContextScorer.score_batch` scores a whole
  candidate set in one pass — the per-component scores are reduced against the
  weight vector together (a single matrix product under NumPy via the new
  `vincio.context.vectorized`, an identical pure-Python reduction otherwise), and
  each `ContextScores` is built without per-item validation. Bit-for-bit
  identical selection to the per-candidate loop. VincioBench gates
  `families.perf.vectorized_scoring.equivalent`.
- **Compiled-prompt render program.** `PromptCompiler` compiles a spec's stable
  prefix (role/objective/rules/safety/definitions/output-contract/examples) once
  into a reusable render program (`vincio.prompts.program`,
  `CompilerOptions.use_render_program`, default on) and reuses it across calls
  that share the spec, rendering only the volatile suffix. Byte-identical output;
  `program_hits` counts reuses. VincioBench gates
  `families.perf.render_program.byte_identical`.
- **Warm candidate arena.** When the candidate set (inputs + privacy scope) is
  unchanged, the context compiler reuses the collected, normalized, and
  privacy-screened candidates (`vincio.context.arena`,
  `performance.reuse_candidate_set` / `ContextCompilerOptions.reuse_candidate_set`,
  default on) instead of rebuilding them. Correctness-preserving and safe under
  concurrent use; `arena_hits` counts reuses. VincioBench gates
  `families.perf.warm_arena.equivalent`.
- **Streaming-first compilation.** `ContextCompiler.compile_streaming` yields a
  new `CompileStreamEvent` stream — the stable prefix (objective / instructions /
  constraints / task) before any candidate is scored, then the selected evidence,
  then a terminal `done` carrying the full `CompiledContext` (identical to
  `compile`). Back-pressure is the async generator itself. VincioBench gates
  `families.perf.streaming_compile.prefix_before_scoring`.
- **Speculative retrieval prefetch.** Opt-in `performance.speculative_prefetch`
  warms the query embedding (`vincio.retrieval.SpeculativePrefetcher` /
  `PrefetchHandle`) from the task classification while preparation runs, so
  retrieval's query embed lands as a cache hit; cancelled cleanly and best-effort.
  VincioBench gates `families.perf.prefetch.warms_cache`.
- **Per-app memory-footprint budget.** `performance.memory_budget_mb` declares a
  resident-memory ceiling for the compiled packet; the compiler slims the packet
  and evicts the lowest-utility evidence to fit (`vincio.context.footprint`),
  recording each eviction. The footprint is surfaced as `RunResult.memory_bytes`
  and rolled up as `peak_resident_bytes` in the cost summary. VincioBench gates
  `families.perf.footprint.budget_enforced` and a resident-footprint regression
  gate `families.perf.footprint.packet_bytes`.

### Performance

- New SLOs: p99 cold-compile latency, a sub-millisecond warm-compile hot path
  (`families.perf.context_compile.cached_p50_ms`), and a resident-footprint
  ceiling, plus the equivalence/byte-identity/streaming/prefetch invariants —
  each backed by an at-least-as-strict VincioBench budget.

**1663 tests passing offline; ruff + mypy clean; VincioBench 287 budgets / 96 SLOs.**

## [3.0.1] - 2026-06-18

Closes the two honest scoping notes the 3.0.0 milestone shipped with. Both
additive and backward-compatible — `API_VERSION` stays `3.0`, and the default
`app.deploy(dataset=...)` and `run`/`arun`/`abatch` paths are unchanged.

### Added

- **Live-traffic canary bound to the deploy surface.** `app.deploy` (and
  `deploy_candidate`) gain a live mode — `app.deploy(candidate, live_inputs=...,
  score_fn=...)` — that ramps `CanarySpec.percent` of the supplied live runs onto
  the candidate prompt/policy, scores each arm online with `score_fn(RunResult)`,
  and once `min_samples` candidate observations land applies the same
  no-regression verdict: promote, or **freeze + auto-roll-back** on a regression.
  The new `LiveCanary` (`vincio.optimize`) is the reusable prompt-layer analog of
  the 1.8 `CanaryRouter` (per-run observation via `aobserve`, `verdict()`,
  `afinalize()`); each observation still returns a real answer to the caller.
  `CanarySpec` gains `percent`. VincioBench gates
  `families.loop.self_improvement.live_canary_promotes` / `live_canary_rolls_back`.

### Changed

- **The async-canonical run path is now literally true on every path.** The
  batch path's `VincioRuntime._persist_run` is now a coroutine that persists
  through the canonical async store contract (`await asave`), matching the
  interactive/streaming epilogue — so no run path blocks the event loop with a
  synchronous store write. VincioBench gates
  `families.scale.async_canonical.run_path_persists_async`.

**1623 tests passing offline; ruff + mypy clean; VincioBench 277 budgets / 87 SLOs.**

## [3.0.0] - 2026-06-18

The breaking culmination — fewer, truer abstractions. 3.0 is the second
deliberate breaking window (after 2.0): it unifies the 2.x self-improvement
organs under one declarative contract, makes erasure **provable** with consent
modeling, and makes the async store/event contracts canonical. `API_VERSION`
moves to `3.0` and `EVENT_SCHEMA_VERSION` to `3.0`. Nothing breaks *outside* the
window — the flat `app.<method>` API, the 2.x organs, and every existing run path
stay fully supported; the new surface is `@experimental(since="3.0")`.

### Added

- **Unified declarative self-improvement contract** (`vincio.optimize.self_improvement`).
  One `SelfImprovementPolicy` composes scheduling, autonomous proposal, online
  updates, canary/rollback, active-learning label acquisition, and
  meta-optimization. `app.self_improvement(policy, dataset=...)` returns a
  `SelfImprovementController` whose `astream()` / `step()` / `run()` drive the
  existing `ImprovementLoop`, `ExperimentProposer`, `ContinuousImprovementController`,
  and canary as **one streaming engine**, emitting `observe → proposal → meta →
  label → reeval → canary → promote/rollback` events on the shared audit chain and
  event bus. Meta-optimization ships as `successive_halving` (over the
  strategy/budget grid) + `learn_fitness_weights`; active learning as
  `select_for_labeling`. Every promotion still passes the same significance +
  safety + golden non-regression gates.
- **Canary-gated deployment** — `app.deploy(candidate, dataset=...)` /
  `deploy_candidate` promote a prompt/policy live (registry push + tag + apply +
  audit) only on a no-regression `CanaryVerdict`, and refuse + roll back to the
  last known-good version otherwise. This is the canary-driven promotion surface
  reserved out of 1.10.
- **Provable erasure** — `app.erase_source(...)` now returns a signed,
  content-bound `ErasureProof` on `ErasureResult.proof`: a manifest of exactly
  which chunk / document / memory / **generated-artifact** ids were removed, bound
  by SHA-256 over the sorted removed-id set, signed with the app's
  `content_signer`, and anchored to the audit chain's Merkle root
  (`build_erasure_proof` / `verify_erasure_proof`). `LineageRecord` gains
  `artifacts` + `LineageIndex.record_artifact`, so an erased source is erased as
  evidence, memory, *and* generated output in one operation.
- **Consent & purpose modeling** (`vincio.governance.consent`) — a `ConsentLedger`
  binds a data subject to a GDPR `Purpose` and `LawfulBasis`
  (`grant` / `revoke` / `check`), persisted to the store and audited.
  `app.use_consent_ledger()` wires it into `AccessController.check_purpose` and
  memory recall, which drops any item whose purpose lost consent. `AccessDecision`
  carries `purpose` / `lawful_basis`.
- **Bi-temporal, ACL-gated memory** — `MemoryItem` gains `valid_from` / `valid_to`
  (+ `valid_at()`), a per-memory `acl` (+ `readable_by()`), and `purpose` /
  `consent_id`. `MemoryScope.TEAM` and `MemoryEngine.for_team(...)` add team-shared
  memory; `MemoryEngine.correct(...)` closes a fact's valid interval and opens a
  corrected one; `recall` / `asearch` accept `as_of=` (as-of recall, including
  superseded facts), `reader=` (ACL), and `team_id=`. SQLite persists the new
  columns and migrates a pre-3.0 store in place.
- **Async-canonical core & finalized telemetry** — `InMemoryMetadataStore` is now
  async-native (`asave` / `aget` / `aquery` / `adelete` / `acount`), so the
  module-level helpers take the native fast path with no worker-thread hop. The
  typed event catalog gains `SelfImprovementPhaseEvent`, `DeployCompleted`, and
  `SourceErased`; `EVENT_SCHEMA_VERSION` is `3.0`.
- `examples/38_self_improvement_and_provable_erasure.py`, the VincioBench `loop` /
  `governance` / `scale` / `memory` family checks for the above, eight new SLOs
  (**274 budgets, 85 SLOs**), and a runnable example smoke-tested offline.

### Changed

- `API_VERSION` → `3.0`; `vincio.__version__` → `3.0.0`.
- The public surface adds `SelfImprovementPolicy`, `SelfImprovementController`,
  `CanarySpec`, `DeployResult`, `ErasureProof`, `verify_erasure_proof`,
  `ConsentLedger`, `Purpose`, and `LawfulBasis` (plus the `vincio.optimize` /
  `vincio.governance` subpackage exports).

### Deprecated

- `app.continuous_improvement(...)` and `app.experiment_proposer(...)` are
  deprecated (`since=3.0`, `removed_in=4.0`) in favour of `app.self_improvement`.
  Both stay fully functional through the 3.x line; the underlying
  `ContinuousImprovementController` / `ExperimentProposer` classes remain public.

**1613 tests passing offline in ~7s; ruff + mypy clean.** The 3.0 milestone
carries no deferred items.

See [ROADMAP.md](ROADMAP.md) for the milestone framing.

## [2.2.1] - 2026-06-18

Closes the two honest scoping notes the 2.2.0 milestone shipped with. Both
additive and backward-compatible — `API_VERSION` stays `2.0`, and the default
`run` / `arun` / `replay` paths are unchanged.

### Changed

- **Token streaming is now genuine provider-driven streaming.**
  `AgentExecutor.astream` (and, through it, `Crew.astream`) route the
  answer-producing model calls through `provider.stream()`, emitting the
  provider's **real token deltas** as they arrive and reconstructing the final
  `ModelResponse` from the stream's `done` event — replacing the 2.2.0 post-hoc
  word-grouping of the finished text. Structured-output (schema) calls stay on
  `generate` (their JSON is not emitted as user-facing text). For the
  deterministic `MockProvider` this surfaces as real 16-char chunk deltas
  offline; for hosted providers it is true token-by-token streaming. VincioBench
  gates `families.agent.streaming.genuine_token_streaming` / `provider_deltas`.

### Added

- **A live-run path for the benchmark adapters — the identical scorer on fresh
  agent output, not just recorded replay.** `adapter.run(solver)` solves each
  task live and scores it with the same `score()` as `replay()`. `make_agent_solver`
  turns a `ContextApp` / `AgentExecutor` (or any callable) into a solver
  (`mode="text"` for an answer; `mode="calls"` captures the agent's function calls
  from its event stream for BFCL), and `make_env_solver(policy)` runs a policy
  through the τ-bench world. Official task sets load with `tasks_from_jsonl` and
  the per-benchmark `gaia_tasks_from_export` / `swebench_tasks_from_export` /
  `bfcl_tasks_from_export` (which parse the released field names, including
  SWE-bench's JSON-encoded `FAIL_TO_PASS` / `PASS_TO_PASS`). VincioBench gates
  `families.agentic_evals.environment_eval.adapters.live_run_scored`, exercised
  end to end offline against a real `AgentExecutor` (the agent genuinely calls
  tools; the BFCL AST scorer grades the calls it made). **258 budgets, 77 SLOs.**

## [2.2.0] - 2026-06-18

Prove it on the world's benchmarks: environment eval, agentic leaderboards, the
governed agent fabric, and generative UI. Entirely additive behind
`@experimental(since="2.2")` on the frozen 2.0 surface — `API_VERSION` stays
`2.0`, the single-process asyncio path stays the default, and nothing here is
required to run Vincio. All offline and deterministic; the benchmark adapters and
registry clients use only the core `httpx` dependency.

### Added

- **Stateful-environment eval harness + task-success oracle.** A new
  `vincio.evals.environment` ships an `Environment` protocol
  (`reset` / `step` / `observe` / `verify`), a deterministic in-process
  `ToolEnvironment` (whose world is a dict mutated by tools), a declarative
  end-state oracle (`StateCheck` / `TaskVerification`), and an
  `EnvironmentSimulator` that drives an agent *policy* through a *mutable* world
  and projects the interaction onto the existing `Trajectory` — scoring
  **verifiable end-state**, not turn-by-turn plausibility. `make_retail_environment`
  is a τ-bench-style reference world; `scripted_policy` / `task_success` round it
  out. Re-exported from `vincio` (`Environment`, `ToolEnvironment`,
  `EnvironmentSimulator`, `make_retail_environment`).
- **Agentic benchmark adapters (SWE-bench Verified / τ-bench / τ²-bench / GAIA /
  WebArena / BFCL).** `vincio.evals.benchmarks` ships one `BenchmarkAdapter`
  contract and the five adapters, each scoring the benchmark's own **verifiable
  end state** (SWE-bench's fail-to-pass/pass-to-pass transition, τ-bench's database
  end state via the environment oracle, GAIA's normalized exact match, WebArena's
  functional check, BFCL's AST match). Each pins its task set by a content hash
  (`task_set_hash()`, verified against the fixture on load) and degrades to
  recorded-fixture replay offline (`adapter.replay()`; fixtures in
  `benchmarks/fixtures/`); `BenchmarkReport.to_eval_report()` projects onto an
  `EvalReport` the Pareto optimizer consumes. `load_benchmark` / `available_benchmarks`.
- **Retrieval evaluation harness + index-version regression.**
  `vincio.evals.retrieval_eval` (`RetrievalEvaluator` / `RetrievalGoldenSet` /
  `RetrievalConfig`) benchmarks an embedder / reranker / chunker / index config on
  recall@k / nDCG@k / MRR / context-precision (reusing the retrieval metrics), and
  `retrieval_regression(...)` records a versioned artifact and gates a recall/nDCG
  regression on **the same significance test as a model swap** (`ab_test`).
  Artifacts persist through `vincio.storage.index_regression`
  (`IndexRegressionStore` / `IndexRegressionArtifact` / `config_key`), keyed on
  `(embedder, chunker, corpus hash)` over the `MetadataStore`.
- **The governed agent fabric (AGNTCY / ACP + MCP Registry).** `vincio.registry`
  ships an `AgentDirectory` (`AgentRecord` / `AgentResolution`) over the existing
  A2A Agent Card — `find` by capability/tag/query, `resolve` governed by an
  allow-list and recorded as an `agent_resolve` access decision on the audit chain.
  An **AGNTCY/ACP** (REST-native Agent Connect Protocol) adapter (`ACPClient` /
  `ACPAgentManifest` + `acp_to_agent_card` / `agent_card_to_acp`) and an **MCP
  Registry** discovery client (`MCPRegistryClient` / `MCPServerRecord`) discover
  agents/servers into the same directory under the same allow-list. A new
  `AllowListGate` (`vincio.security.access`) is a fail-closed reachability gate over
  `AccessController`; `app.agent_directory(allow=..., deny=...)` builds a directory
  wired to the app's audit chain. Re-exported from `vincio` (`AgentDirectory`,
  `AllowListGate`).
- **Generative UI / AG-UI streaming.** `vincio.server.agui` ships an AG-UI /
  MCP-UI compatible event protocol (`AGUIEvent` / `AGUIEventType`) and translators
  (`run_stream_to_agui`, `agent_stream_to_agui`, `agui_sse`), plus the SSE endpoint
  `POST /v1/apps/{app_id}/agui`. `AgentExecutor.astream(...)` and `Crew.astream(...)`
  now yield flat `AgentEvent` / `CrewEvent` streams (run/step lifecycle, real text
  deltas, `tool_call` / `tool_result`, a terminal `done` carrying the state/result)
  matching the `graph` / `compose` streaming surface — crew streams forward each
  member's tool/text events. `mcp.MCPUIResource` (`from_html` / `from_agui`) serves
  MCP-UI resources via `build_app_server(..., ui_resources=[...])` /
  `app.serve_mcp(ui_resources=[...])`. The interactive UI inherits the run's
  provenance, budget metering, and audit — one streamed run.
- **VincioBench guarantees.** New CI-gated checks fold into the existing families:
  environment task-success oracle + benchmark-adapter determinism in
  `agentic_evals.environment_eval`, retrieval-eval recall/nDCG + index-version
  regression in `rag.retrieval_eval`, the governed fabric (AGNTCY/ACP + MCP-registry
  discovery under the allow-list, audited resolution) in `protocols.fabric`, and
  token/tool-event + AG-UI streaming in `agent.streaming` — 255 budgets, 77 SLOs.
  New runnable example `37_benchmarks_and_fabric.py`.

### Notes

- Backward-compatible and additive: every new symbol is `@experimental(since="2.2")`
  and reachable through a new entry point; no existing API changes behavior. The
  benchmark adapters and reference environments are offline and deterministic and
  never reach the network; the agent fabric is governed by construction (fail-closed
  allow-list, every resolution audited); AG-UI streaming opens no new data-exposure
  boundary (it is a translation of the run's existing `astream`).

## [2.1.1] - 2026-06-18

Closes the three known limitations the 2.1.0 adversarial review surfaced. All
additive and backward-compatible — `API_VERSION` stays `2.0`, the single-process
asyncio path stays the default, and no existing graph, reducer, or backend
changes behavior.

### Added

- **Channel-default reducers — a map-reduce no longer needs a seed node.**
  `StateGraph(..., defaults={...})` (and non-required `state_schema` field
  defaults, inferred automatically) declare a reduced key's empty value, so the
  reducer folds the **first** write into that default instead of passing the raw
  value through. A `Send` map-reduce can now use a non-defensive reducer
  (`operator.add`) with no upstream node seeding the collected key. The legacy
  first-write passthrough is unchanged whenever no default is known, so existing
  bare-callable reducers keep their exact semantics. Defaults ride through
  `app.graph(defaults=...)` and survive `RayBackend` export. This replaces the
  2.1.0 workaround of seeding the collected key, at its root.
- **`vincio.testing.assert_backend_conformance`** (+ `conformance_cases`) — the
  offline contract every runtime backend must satisfy: it runs a battery
  (sequential, conditional routing, `Send` map-reduce with a channel default)
  through a backend and asserts it reproduces the native durable engine. The
  `RayBackend` / `TemporalBackend` export adapters — which can only be exercised
  against injected fakes offline — are now held to this contract, not merely
  "runs one graph," and a real cluster wiring can validate itself the same way.
  VincioBench's scale family gains `backend_conformant` and
  `map_reduce_no_seed_ok` budgets.

### Changed

- **The real local-neural-model paths are now exercised offline.**
  `SpladeEncoder`, `LocalCrossEncoderReranker`, and `FastEmbedEmbedder` accept an
  injected model object (`model=` / `tokenizer=` / `torch_module=`), mirroring
  `GGUFProvider(llama=...)`, so the real forward / `predict` / `embed` paths run
  against faithful fakes with the heavy deps absent. `SpladeEncoder.pool_logits`
  extracts the SPLADE log-saturated max-pool + top-k into pure, directly tested
  Python (the model forward stays in `torch`). The `# pragma: no cover` markers
  on those real-model paths are removed — they are covered now.

### Notes

- 1497 tests passing offline; ruff + mypy clean. VincioBench: 18 families, 231
  CI budgets, 71 SLOs. The 2.1 milestone now carries no deferred items.

## [2.1.0] - 2026-06-17

Scale out & train for real — distributed execution, executed fine-tuning, and a
served (still self-hosted) observability plane. Entirely additive behind
`@experimental` on the frozen 2.0 surface; `API_VERSION` stays `2.0`, the
single-process asyncio path stays the default, and nothing here is required to
run Vincio.

### Added

- **Distributed durable-execution backend.** `vincio.agents.distributed` adds a
  `GraphCoordinator` protocol (in-memory `InMemoryGraphCoordinator` +
  `RedisGraphCoordinator`) and a `DistributedCheckpointer` that lease-guards
  each graph thread (a TTL `running` lease) and CAS-commits every super-step
  (checkpoint-version optimistic concurrency), so two workers can never
  double-execute a step — the loser raises `CheckpointConflictError`. New
  runtime backends in `agents/backends.py`: `WorkerPoolBackend` (the in-process
  reference distributed executor, with `run_batch` fan-out) plus `RayBackend`
  and `TemporalBackend` export adapters (lazy/injectable, offline-testable). The
  durable graph gains true BSP parallel super-steps (`StateGraph.compile(parallel=True)`)
  and a `Send` primitive for map-reduce fan-out; `Workflow.map_step` adds
  data-dependent level-parallel spawning. Lease/CAS metadata rides the same
  checkpoint records, so a thread moves between the single-process and
  distributed backends without losing its ledger or trace.
- **Executed distillation & provider fine-tune jobs.** `vincio.providers.finetune`
  ships `OpenAIFineTuneBackend`, `GoogleFineTuneBackend`, and
  `AnthropicFineTuneBackend` (submit/poll/cancel) plus `run_finetune` and a
  `make_finetune_backend` factory. `optimize.provider_trainer` turns the
  `StudentTrainer` from an injected no-op into an executed trainer that submits a
  fine-tune job, registers the resulting model in the registry, and returns the
  trained model id; `BootstrapFinetune` gains an optional `swap_gate` so the
  student is promoted only past the significance gate. The export gains
  `semantic_dedupe` and a `max_example_chars` truncation guard. Offline, the job
  lifecycle runs against `httpx.MockTransport` cassettes and the promotion
  decision is fully deterministic.
- **Served observability & alerting plane.** `observability.IndexedTraceStore` is
  an indexed SQLite trace/cost store with time-bucketed cost rollups, retention
  (`purge`), and percentile/cost-by-dimension queries that replace O(n) JSONL
  scans. `observability.ViewerApp` + `serve_viewer` serve a dashboard, live trace
  tail, search, and JSON APIs over it using only the standard library. A new
  `AlertSink` protocol with `WebhookAlertSink` / `SlackAlertSink` /
  `PagerDutyAlertSink` / `PrometheusExporter`, plus an `AlertManager` rule engine
  (`AlertRule`: threshold / EWMA-Welford anomaly / SRE burn-rate) that runs over
  the cost ledger and event bus, and a `TailSamplingExporter` (error-prioritized,
  deterministic). The zero-dependency static viewer stays; this plane is opt-in
  and emits on the same audit chain.
- **Redis-backed shared server state + `vincio serve`.** `storage.shared_state`
  adds `RateLimiter` / `IdempotencyStore` protocols (in-memory defaults +
  `TenantQuotaManager`); `storage/redis.py` adds `RedisRateLimiter` and
  `RedisIdempotencyStore` so multi-worker deployments stay coherent. A first-class
  `vincio serve` launcher (uvicorn) plus `/v1/health/ready` and `/v1/metrics`
  (Prometheus), a lifespan with graceful shutdown, and an optional per-caller
  rate-limit middleware.
- **Content-capture controls.** `observability.ContentCapturePolicy` gates
  prompt/completion content at the export boundary — **off by default** — and
  redacts (PII) + truncates when opted in, before content reaches OTel events,
  JSONL, or the viewer. Wired into the OTel exporter and the tool runtime.
- **Quantization + two-stage retrieval.** `retrieval.quantization` adds
  `quantize_scalar` / `quantize_binary` and a `TwoStageIndex` (coarse search on
  quantized/Matryoshka-truncated vectors, exact rerank on full precision),
  reusing `mrl_truncate`. The Qdrant adapter accepts a native `quantization=`
  config.
- **Batteries-included local neural models** (optional deps, with deterministic
  offline fallbacks): `FastEmbedEmbedder` (ONNX/fastembed dense), `SpladeEncoder`
  (real SPLADE sparse), `ColBERTTokenEmbedder` (late-interaction tokens),
  `LocalCrossEncoderReranker`, and a native llama.cpp `GGUFProvider` with
  on-device embedding. New extras: `vincio[fastembed]`, `vincio[splade]`,
  `vincio[cross-encoder]`, `vincio[gguf]`, `vincio[local-neural]`.

### Quality & release

- **1485 tests passing offline in ~6s; ruff + mypy clean**; thirty-six runnable
  examples; VincioBench gates the 2.1 guarantees under CI budgets (229 budgets,
  71 SLOs): distributed durability + multi-worker shared-state coherence in
  `scale`, the executed-distillation swap-gate in `loop`, quantized two-stage
  recall in `rag`, and burn-rate/EWMA alerting in `cost`.

## [2.0.1] - 2026-06-17

Closes the one deferred 2.0 follow-up and a secret-scanning hygiene issue. No
public-API changes (`API_VERSION` stays `2.0`).

### Changed

- **Native filter pushdown now reaches every named backend.** Pinecone,
  Weaviate, Milvus, and Elasticsearch/OpenSearch persist flat filterable fields
  alongside the chunk blob (`flat_filter_fields`) and pass the compiled
  `FilterSpec` into the backend's native query (Pinecone metadata filter,
  Weaviate `where`, Milvus `expr`, ES/OpenSearch kNN `filter`), so tenant /
  document / kind / metadata scope is applied server-side — not only client-side.
  Each is verified offline against its fake (which now applies the pushed-down
  filter). The shared-or-mine tenant scope matches both null (in-memory) and the
  empty-string-stored untagged case so it is correct in-memory and natively.
  `PineconeVectorIndex` now lazy-imports its SDK only when building a real client
  (consistent with the other adapters), so an injected client works without the
  package.
- **Secret-scanning hygiene** — the synthetic OpenAI-key fixture used to exercise
  the egress DLP scanner (tests, example, benchmark) is now assembled at runtime,
  so no contiguous secret-shaped literal lives in source. It still trips the
  `sk-...` detector at scan time, which is the point of the test.

### Notes

- 1389 tests passing offline; ruff + mypy clean. VincioBench: 18 families, 218
  CI budgets, 65 SLOs. The 2.0 milestone now carries no deferred items.

## [2.0.0] - 2026-06-17

The one breaking window. Five milestones of additive growth exposed structural
debt the frozen 1.0 surface could not pay down. 2.0 is the single deliberate
breaking release — nothing breaks outside it — and it lands the flagship
multimodal-native Context Packet that genuinely needs the schema change. The
public-API contract (`API_VERSION`) moves to `2.0`.

### Added

- **Capability facades** — `ContextApp`'s surface is decomposed into six narrow,
  lazily-constructed, independently-testable views (`vincio.core.facades`):
  `app.runs` / `.knowledge` / `.governance` / `.optimization` / `.serving` /
  `.training`. Each exposes one cohesive method group and delegates to the app's
  implementation; reaching across a boundary raises `AttributeError`. Built on
  first access, so cold start and footprint scale with what an app uses.
- **Multimodal-native Context Packet** — `EvidenceItem` and `ContextCandidate`
  generalize from text-only to typed `modality` (`text` / `image` / `table`)
  with `image` (`ImageRef`) / `table` carriers and modality-aware token cost, so
  the compiler selects, budgets, orders, and cites image and table evidence in
  the same scored packet as text. Slim packets are backed by a content-addressed
  evidence store (`vincio.context.evidence_store`: `InMemoryEvidenceStore` /
  `BlobEvidenceStore`) so `ContextPacket.materialize(store=...)` recovers text
  after cross-process deserialization. The evidence ledger gains entailment
  `supports` / `contradicts` links (`link_entailments`).
- **Structured `FilterSpec`** (`vincio.retrieval.filters`) — a declarative,
  serializable filter (`eq` / `ne` / `in_` / `range_` / `exists` / `contains`
  over `and_` / `or_` / `not_`) compiled to each backend's native filter (Qdrant
  `Filter`, pgvector GIN-indexed `jsonb` `WHERE`, Pinecone, Weaviate, Milvus,
  Elasticsearch). Qdrant and pgvector push down server-side and fetch exactly
  `top_k`, fixing the over-fetch under-fill bug; `app.tenant_filter` returns a
  pushdown `FilterSpec` (shared-or-mine), closing the cross-tenant
  fetch-to-filter exfiltration risk.
- **Enterprise endpoints behind a pluggable `AuthStrategy`** — AWS Bedrock
  (pure-stdlib SigV4 Converse), Google Vertex (regional service-account OAuth),
  and Azure OpenAI (deployment routing + `api-version`), registered as
  `bedrock` / `vertex` / `azure` through the same `ProviderRegistry`, capability
  guards, swap gate, residency, and audit chain as every other provider.
- **Async-first storage + typed event catalog + unified telemetry** — an
  `AsyncMetadataStore` protocol with `aget` / `adelete` / `acount` alongside
  `asave` / `aquery` (native async or threaded shim) and a psycopg3
  `AsyncConnectionPool` Postgres fast path; a typed, versioned event catalog
  (`vincio.core.events`: `EVENT_CATALOG`, Pydantic payload models, `publish()`,
  `EVENT_SCHEMA_VERSION`); and one trace fanned out to spans **and** OTel metric
  histograms under the GenAI **agentic** conventions (`invoke_agent`,
  `gen_ai.agent.*`, `gen_ai.usage.cost`).
- **Mandatory egress DLP + signed audit chain** — `PolicyEngine.scan_egress`
  scans the fully-assembled provider request (system + messages + tool schemas)
  at both provider-dispatch boundaries regardless of call-site wiring
  (`security.egress_dlp`: `off` / `warn` / `block`); the hash-chained audit log
  gains per-entry HMAC/Ed25519 signatures and Merkle-root checkpoints
  (`security.audit_signing_key`), making it tamper-evident against a privileged
  attacker who can recompute the public hashes.

### Changed (breaking)

- **Eval metric semantics** — unscoreable cases (no ground truth, no claims, no
  trajectory) return `MetricResult(skipped=True)` and are excluded from gate
  aggregation instead of a neutral `1.0` that inflated means and silently passed
  gates. The lexical metric formerly named `semantic_similarity` is renamed to
  its true identity `lexical_overlap`; `semantic_similarity` is now a real
  embedding-backed metric (configurable via `set_semantic_embedder`).
- **`Index.search` `where` type** widens to `Where = FilterSpec | SearchFilter`;
  the `MetadataStore` async methods are the canonical contract.
- **`HTTPProvider` auth** is refactored behind `AuthStrategy` (`_prepare`); the
  audit-entry schema gains `signature` / `key_id` and `verify` validates them.
- `API_VERSION` → `2.0`; `EvidenceItem` / `ContextCandidate` carry `modality`.

### Notes

- 1386 tests passing offline in ~5s; ruff + mypy clean. VincioBench: 18 families,
  217 CI budgets, 65 SLOs. Thirty-five runnable examples. Every change is retired
  or introduced through the mechanical deprecation runway 1.0 established; the
  flat `app.<method>` API remains fully supported alongside the facades.

## [1.10.0] - 2026-06-17

The loop closes itself. Vincio could already *measure* drift and run an offline
optimizer, but the online loop only closed when a human pressed go. 1.10 makes
self-improvement continual, online, and safe — and opens the agentic frontier
(deep research, self-editing memory, computer-use) on the same cited, grounded,
audited spine. Everything is additive behind `@experimental` entry points on the
frozen 1.0 API; the canary-driven prompt/policy promotion that needs a new
serving surface stays reserved for 2.0.

### Added

- **Online improvement controller** — `app.continuous_improvement(...)`
  (`vincio.optimize.controller.ContinuousImprovementController`) subscribes to
  `drift.detected` + `eval.online`, streams online scores into a CUSUM
  changepoint detector, and turns a *sustained* signal into one of three gated
  actions: a targeted re-eval, a fresh `ImprovementLoop` run, or a rollback to
  the last known-good `prompts/registry.py` version. Per-trigger cooldown
  debouncing and a global eval budget bound it; every trigger, debounce,
  decision, and rollback lands on the hash-chained audit log and an event. State
  (budget spent, sustain counts, cooldowns) persists to the shared store, so the
  controller is restart-safe.
- **Distributional drift + CUSUM** — `evals/drift.py` gains two-sample
  Kolmogorov–Smirnov (`ks_statistic` / `ks_drift`), Population Stability Index
  (`psi`), RBF Maximum Mean Discrepancy (`rbf_mmd2`), and a streaming
  `CUSUMDetector`; `DriftMonitor.observe_score` feeds online scores into a
  per-metric CUSUM that fires `drift.detected` on a sustained shift (the event
  the controller acts on), with restart-safe persisted accumulators.
- **Restart-safe, worker-aggregatable online state** — `OnlineEvaluator`
  persists its 1-in-N sampling counter to the store (`online_state`) keyed by
  `worker_id`; `observed_total()` aggregates across workers.
- **Real provider-backed reflective optimizer (GEPA proper)** — `LLMReflector`
  (`optimize/reflective.py`) wired to the app's own provider reads the *actual*
  failing cases (input + output + expected + grounding), clusters them into
  failure modes (`cluster_failures`), and proposes targeted edits validated
  against the existing edit schema. `HeuristicReflector` stays the air-gapped
  deterministic fallback; `app.reflective_optimize(..., reflector="llm")` and
  `ImprovementLoop(reflector="llm")` opt in. Feeds the same Pareto frontier and
  gated promotion.
- **Autonomous experiment proposer** — `ExperimentProposer` /
  `app.experiment_proposer(...)` ranks where the system is weakest from online
  eval + drift and proposes/schedules the highest-ROI experiment (prompt /
  retrieval / budget / routing / distillation) under a global eval budget, every
  decision recorded.
- **Guarded online bandits** — a contextual `LinUCB` joins `EpsilonGreedyBandit`
  / `UCB1Bandit`, wired into the live route by `GuardedBanditRouter` (a
  `ModelProvider`) behind a **safety floor** (never explores on safety-/high-risk
  traffic), with persisted arm stats, cumulative regret, and auto-freeze /
  rollback-to-safe-arm on regression. `app.use_bandit_router(...)`.
- **Held-out, growing golden regression suite** — `GoldenRegressionSuite`
  (`evals/datasets.py`) records the cases each promotion fixes with provenance
  and gates every later promotion by replay, so sequential auto-promotions can
  never silently undo a prior fix; wired into `ImprovementLoop(golden_suite=...)`.
- **Deep-research agent** — `ResearchAgent` / `app.research(...)` loops
  search → read → reflect → verify → synthesize over the query-understanding
  planners and the grounded-fact extractor under explicit breadth/depth/source/
  token budgets, dedups sources, verifies with judges, and emits a cited report
  through the 1.9 `CitedReportBuilder` — every claim cited and grounded by
  construction, scored for citation coverage / grounding / source diversity.
- **Agent memory OS** — `MemoryOS` / `app.enable_memory_os(...)` exposes
  self-editing memory as permissioned, audited tools (`memory_append` /
  `memory_replace` / `memory_search` / `memory_archive`) over the existing
  guarded write pipeline, with a context-pressure pager between in-context core
  memory and the archival store.
- **In-loop context compaction** — `agents/compaction.py` `ContextCompactor`
  folds old tool/observation turns into a rolling extractive summary at a token
  budget, replacing the fixed `[-8]`/`[:24]` slicing in `agents/executor.py`
  (DAG and ReAct paths), keeping tool-call pairs intact.
- **Level-parallel agent DAG + `plan_and_execute`** — the executor runs each
  topological level's independent steps concurrently (bounded), and
  `Planner.replan` drives a real plan → execute → observe → replan loop for the
  `plan_and_execute` mode.
- **Computer-use / agentic browsing** — `tools/computer_use.py` adds a
  navigate / click / type / screenshot action vocabulary with a deterministic
  `MockComputerUse`, a `PlaywrightComputerUse` backend, and a provider-native
  adapter, exposed via `app.enable_computer_use(...)` as permissioned, audited,
  approval-gated tools.
- **Pluggable isolation backends** — `tools/sandbox.py` gains an
  `IsolationBackend` interface with `Subprocess` (zero-dep default, not a
  security boundary), `Container`, `gVisor`, `microVM`, and `WASM` backends;
  `require_real_isolation` enforces that code-executing and computer-use
  workloads run behind a real boundary.
- **Provider-native hosted tools** — `providers/hosted_tools.py` surfaces OpenAI
  Responses built-ins (`web_search` / `file_search` / `code_interpreter` /
  `computer_use`) as namespaced, permissioned Vincio tools
  (`app.use_hosted_tools(...)`); the Responses adapter emits each as its
  built-in descriptor. `computer_use` is approval-gated.
- New error `SandboxError`; new optional extra `vincio[computer-use]`
  (Playwright); `examples/34_continual_loop_and_agentic_frontier.py`.

### Notes

- 1304 tests passing offline in ~5s; ruff + mypy clean. VincioBench: 17 families,
  205 CI budgets, 60 SLOs. Thirty-four runnable examples.

## [1.9.1] - 2026-06-17

Closes the two thin spots in the 1.9 generation surface so the milestone carries
no deferred follow-ups. Additive and `@experimental`; no public symbol removed.

### Changed

- **Forms Document-AI cloud adapters are now real, dependency-injected
  implementations** instead of `NotImplementedError` stubs. `TextractDocumentAI`,
  `AzureDocumentAI`, and `GoogleDocumentAI` take the SDK client you build and run
  the real `analyze_document` / `begin_analyze_document` / `process_document`
  calls in a worker thread; the response→`FormField` parsing (key/value text,
  confidence, page, and a bounding box) is a pure `parse(...)` function tested
  offline against synthetic responses (no SDK is a hard dependency).
- **Embedded PNG C2PA credentials are now self-verifying.** `embed_provenance`
  binds the embedded credential to the pre-insert bytes; the new
  `extract_embedded_manifest` / `verify_embedded_manifest` reconstruct the
  original bytes by removing the `c2pa.manifest` chunk and confirm the digest, so
  an extracted credential is independently verifiable against the file it travels
  in (a tampered asset fails). The sidecar / returned manifest still bind the
  final bytes.

### Added

- Offline tests for each cloud Document-AI parser, the self-verifying embedded
  credential (incl. tamper rejection), and the optional-dependency error
  messages (PPTX render / Parquet / `.msg` raise a clear install hint when the
  extra is absent). A `families.generation.media.embedded_self_verifies`
  VincioBench budget.

### Notes

- 1196 tests passing offline; ruff + mypy clean. VincioBench: 17 families,
  173 CI budgets, 51 SLOs.

## [1.9.0] - 2026-06-17

Documents & images flow OUT — cited, governed, eval-gated artifacts. Vincio could
read a DOCX, a PDF, and a scanned packet and validate a JSON answer, but stopped
one step short of the deliverable. 1.9 closes the documents-/images-out loop: a
document-generation engine, cited-report assembly, image-generation/editing and
TTS as first-class output modalities, OCR/transcript/figure inputs, new-format
loaders, and an EU AI Act conformity pack — every produced asset cited,
provenance-stamped, budget-metered, and audited on the same chain as text.
Entirely additive behind a new `vincio.generation` subpackage, new `vincio[...]`
extras, and `@experimental` markers on the frozen 1.0 API; no public symbol
removed or repurposed.

### Added

- **`vincio.generation` document engine.** `DocumentBuilder` turns a *validated*
  result (an `OutputContract` output, a `RunResult`, a structured mapping, or
  Markdown) into rendered artifacts — Markdown/HTML dependency-free, DOCX
  (`vincio[gen-docx]`), PDF (`vincio[gen-pdf]`), PPTX (`vincio[gen-pptx]`) — via a
  format-neutral `DocumentModel` IR. Because the input already passed validation,
  the document is grounded by construction. Structural `DocumentContract`
  (required sections, `TableSpec` column specs, length bounds, citation-per-
  section) validates the result with **formatting-only repair** (`repair_formatting`)
  mirroring the JSON-repair path; every render records a `document_generate` audit
  event with the source evidence ids. Adds template/form filling
  (`fill_text_template` / `fill_docx_form` / `fill_pdf_form`, typed citation-aware
  `Slot`s) and `generate_redline` (tracked-change DOCX, `**ins**`/`~~del~~` text).
- **`CitedReportBuilder`.** Resolves inline `[E1]`-style markers to numbered
  footnotes/endnotes and a generated bibliography with per-claim provenance,
  computes sentence-level **citation coverage**, and optionally verifies
  **per-claim entailment** (pluggable backend; strict lexical+numeric default).
  A `CitationContract` enforces a coverage floor, rejects unresolved markers, and
  gates on entailment — replacing the flat "one valid citation anywhere" check.
  New `citation_coverage` and `claim_entailment` eval metrics.
- **Image generation/editing provider abstraction.** `ImageProvider` with
  `generate_image` / `edit_image` / `variation`, a neutral `ImageGenRequest` /
  `ImageGenResponse`, backends for OpenAI `gpt-image-1`, Gemini/Imagen, and a
  generic HTTP/Replicate adapter, plus a `MockImageProvider` that emits real PNGs
  offline. Every asset auto-attaches a media-aware C2PA manifest bound to its
  bytes, is metered against the budget, and is audited (`image_generate`).
- **TTS / speech-synthesis output modality.** `SpeechProvider` with
  `synthesize_speech`, a neutral `SpeechRequest` (voice/format/speed), backends
  for OpenAI TTS, Gemini TTS, and ElevenLabs/Cartesia, plus a `MockSpeechProvider`
  that emits real WAVs. Audio provenance + budget metering + audit
  (`speech_synthesize`), unified with the realtime audio path.
- **Audio as chat input.** `ContentPart.audio` is now rendered by the OpenAI
  (`input_audio`) and Gemini (`inlineData`) chat providers via a shared
  `core.media.encode_audio_bytes`, activating the already-typed `AudioRef` outside
  the realtime WebSocket path.
- **Media-aware synthetic-content marking.** `mark_synthetic_content` accepts
  `str` *or* `bytes` (binds by SHA-256), marks edits with
  `compositeWithTrainedAlgorithmicMedia`, and records the asset's media type.
  New `embed_provenance` (PNG metadata, dependency-free, with an invisible-
  watermark hook) and `write_sidecar_manifest` (a `*.c2pa.json` for any format).
- **Richer document inputs.** OCR auto-fallback in `load_pdf` (low-text pages
  rasterized + OCR'd, `extractor='ocr'` per page, `vincio[ocr]`); `load_media` for
  audio transcript ingestion via a `Transcriber` protocol
  (`MockTranscriber` / `WhisperTranscriber` / `ProviderAudioTranscriber`);
  `figure_evidence` turning PDF figure crops into citable evidence with bounding
  boxes; a real-parser HTML path (`parse_html`, table extraction) and structured
  JSON/JSONL/YAML (`structure_data`).
- **New format loaders + parser registry.** Dependency-free PPTX/EPUB/RTF/ODT,
  plus Parquet (`vincio[parquet]`), mbox, and `.msg` (`vincio[msg]`), behind a
  unified `ParserRegistry` (`register_loader`) that replaces the if/elif suffix
  chain. Forms/KYC extraction via a `DocumentAI` protocol (Textract / Azure /
  Google adapters) and an offline `HeuristicFormExtractor`, returning `FormField`s
  with confidence (+ bbox) convertible to evidence (`form_fields_to_evidence`).
- **EU AI Act conformity pack.** `RiskTierClassifier` (advisory risk-tier
  placement), `AnnexIVBuilder` (cited Annex IV technical documentation), and
  `FRIAGenerator` (Article 27 fundamental-rights impact assessment) — all
  generated from the live config, cards, compliance matrix, and eval/red-team
  evidence through the document engine, recorded as `conformity_doc` audit
  events (`app.risk_tier` / `app.annex_iv` / `app.fria`). An ISO/IEC 42001
  control catalog joins the `ComplianceMapper` family.
- **App methods** (all `@experimental`, since 1.9): `build_document`,
  `cited_report` / `acited_report`, `generate_image` / `agenerate_image`,
  `synthesize_speech` / `asynthesize_speech`, `load_media`, `risk_tier`,
  `annex_iv`, `fria`.
- **VincioBench `generation` family** + three SLOs and CI budgets covering
  document-contract validity, cited-report coverage + entailment, media-provenance
  binding/disclosure, redline correctness, new-format ingestion recall, and
  generated-media prompt safety. New `examples/33_documents_and_media_out.py`.

### Changed

- `ComplianceFramework` gains `ISO_42001` and `EU_AI_ACT`; `CONTROL_CATALOG` adds
  ISO/IEC 42001 controls (so the compliance matrix now spans five mapped
  frameworks). `ModelCapabilities.output_modalities` is the idiomatic generation-
  capability flag.

### Notes

- 1189 tests passing offline in ~5s; ruff + mypy clean. VincioBench: 17 families,
  172 CI budgets, 51 SLOs. No deferred follow-ups.

## [1.8.1] - 2026-06-17

Closes the two deliberately-scoped follow-ups documented at 1.8.0, so the
milestone carries no deferred items. Additive under the frozen 1.0 API; no public
symbol removed or repurposed.

### Changed

- **Residency is now a run-boundary choke point over *every* reachable model.**
  Previously `app.use_router` / `shadow` / `canary` / `use_cascade` validated their
  candidate models against the residency policy at wiring time, but a run that
  picked a different candidate per request was only checked for the primary model
  at the choke point. `check_residency` now enumerates the full reachable set —
  the configured/per-run model, any budget-degrade target, every cascade rung, and
  the candidates of a `Router` / `ShadowProvider` / `CanaryRouter` wrapper — and
  refuses egress for any disallowed-region model, on the same hash-chained audit
  path. Wiring-time enforcement stays as a fail-fast. A no-op when no residency
  policy is configured (the default), so there is zero overhead otherwise.

### Added

- **Recorded-cassette tests for the `GoogleBatchBackend` wire format.** The full
  Gemini Batch Mode lifecycle (submit → poll → results → cancel) is now exercised
  offline against an `httpx` mock transport returning recorded Gemini-shaped
  responses — asserting the request URL/path, the inlined-request envelope keyed
  by `custom_id`, `BATCH_STATE_*` status mapping, response parsing through the
  provider's own parser, reconciliation, and half-cost billing — so the wire
  handling is verified without a live endpoint. The backend docstring now scopes
  it precisely to the Google Developer API (Vertex AI's service-account + GCS
  batch surface lands with the 2.0 enterprise endpoints).

### Notes

- 1104 tests passing offline; ruff + mypy clean. VincioBench unchanged
  (159 budgets, 48 SLOs), all green.

See the [roadmap](ROADMAP.md) (1.8 milestone).

## [1.8.0] - 2026-06-17

Turns the 1.7 model registry into a **rotation-and-regression discipline** — the
migration safety net for the single most common and riskiest production change, a
model swap. Capability guards refuse to substitute a model that cannot serve the
request; a `SwapGate` replays golden traces and runs an eval + cost + latency +
behavioral diff with statistical backing on every candidate; a shadow provider
and a capped canary qualify a model on live traffic with automatic rollback; and
a lifecycle watcher proposes migrations off deprecated models. Every piece is
pure composition of 1.7 organs (the registry, `ReplayRunner`, `ab_test`,
`DriftMonitor`, `evaluate_gates`, the cost model). All additive behind
`@experimental` entry points on the frozen 1.0 API; nothing changes for callers
who do not opt in.

### Added

- **Capability-aware routing preflight + cost/latency `Router`.** A new
  `vincio.providers.capabilities` module (`requirements_for`, `capability_check`)
  intersects a request's needs (vision, tool calling, structured output,
  reasoning, context length) with a candidate's `ModelCapabilities`. A registry-
  backed `Router` (`vincio.optimize.routing.Router`, also re-exported from
  `vincio`) picks the cheapest / fastest / least-busy *capable* model per request,
  load-balances across equivalents, and **downgrades** to honor a per-request
  budget, emitting a `model.routed` decision. Wire it with `app.use_router(...)`.
- **Capability + lifecycle guard on failover & cascades.** `FailoverChain` and
  `HealthAwareFailover` now (by default, opt out with `guard_capabilities=False`)
  skip a capability-mismatched substitution instead of returning a silently wrong
  answer, classify a **terminal lifecycle/config error** (retired/removed/unknown
  model) distinctly from a transient outage (`is_lifecycle_error`), and surface
  `ModelRetiredError` ("rotate now") when every candidate is retired. The runtime
  cascade starts on, and escalates only into, a capable rung. New errors
  `CapabilityMismatchError` / `ModelRetiredError`. Unknown models are never blocked.
- **`SwapGate` + model-swap regression.** A new `vincio.evals.swap` module:
  `SwapGate` (`app.gate_swap(...)` / `vincio providers regress`) replays golden
  traces and runs `evaluate_gates` + `DriftMonitor` + `ab_test` with behavioral
  shape diffs (tool-call rate, refusal rate, output-length distribution) into a
  PASS/FAIL verdict with p-value and effect size; `model_swap_regression`
  (`app.swap_regression(...)` / `vincio eval regress --baseline-model X
  --candidate-model Y`) holds prompt/data/config fixed, swaps only the model, and
  reports per-metric significance, per-case deltas, the cost/latency trade, and
  the worst-regressed slices.
- **Flake control on `EvalRunner`.** `repeats=N` runs each case N times with
  per-case mean/stdev and configurable `repeat_aggregate`; `flake_quarantine`
  tags noisy cases and excludes them from gate aggregation so non-mock provider
  variance never flips a gate on a single run.
- **Shadow provider + progressive canary with auto-rollback.** `ShadowProvider`
  returns the primary's response while asynchronously dual-dispatching the
  candidate and recording both for offline diff; `CanaryRouter` ramps a configurable
  percentage of traffic to a candidate, scores both arms online, and
  auto-rolls-back to the last known-good model (and prompt-registry head) on
  regression. Both implement `ModelProvider`, so they nest inside `CircuitBreaker`
  / `KeyPool`. Wire with `app.shadow(...)` / `app.canary(...)`.
- **Lifecycle watcher + migration proposals.** `LifecycleWatcher`
  (`app.watch_lifecycle(...)` / `vincio providers lifecycle`) emits early sunset
  warnings and proposes a migration — to a model's declared successor or a cheaper
  Pareto-dominating, at-least-as-capable model — that can rewrite a
  `ModelCascade` / `RoutingPolicy` / `config.model` in place.
- **Live model discovery + Google/Vertex batch parity.** `ModelProvider.list_models`
  (implemented for OpenAI/Anthropic/Google) + `ModelRegistry.reconcile` and
  `discover_models` (`vincio providers discover`) reconcile a provider's live model
  list into the registry offline-safe. A `GoogleBatchBackend` joins
  `providers.batch`, and Google models gain batch-tier pricing, completing
  half-cost batch parity with OpenAI/Anthropic.
- **CLI.** `vincio eval regress` and a new `vincio providers` group
  (`list` / `lifecycle` / `discover` / `regress`).

### Notes

- 1090 tests passing offline in ~4.5s; ruff + mypy clean. Thirty-two runnable
  examples (`examples/32_swap_regression.py` swaps a model end to end through the
  gate and a canary). VincioBench extended in the `reliability`, `cost`, `evals`,
  and `scale` families (159 budgets, 48 SLOs), all green.
- Backward compatible. The one intentional, non-breaking behavior change: failover
  chains guard capabilities by default — they skip a *known-incapable* model and
  try the next capable one rather than attempting a substitution that would drop
  content. Unknown models are never blocked, and `guard_capabilities=False`
  restores the pre-1.8 attempt-everything behavior.

See the [roadmap](ROADMAP.md) (1.8 milestone).

## [1.7.1] - 2026-06-17

Closes the one documented 1.7 known limitation: the intermittent
`test_improvement_loop_reflective_promotes` flake. Additive under the frozen 1.0
API; no public symbol removed or repurposed.

### Fixed

- **Reflective optimizer honors `FitnessWeights` when building its Pareto
  frontier.** `ReflectiveOptimizer` accepted `weights` but always selected over
  the full `DEFAULT_OBJECTIVES`, so an axis the caller weighted to `0.0` still
  reached multi-objective selection. For wall-clock `latency`, that let timing
  jitter flip the knee point between otherwise-tied candidates — the root of the
  intermittent `test_improvement_loop_reflective_promotes` failure (it surfaced
  hash-seed/ordering-sensitively at the frontier-selection step). A new
  `objectives_from_weights()` helper derives the frontier axes from the weights
  (dropping zero-weighted axes, and tracking the configured `accuracy_metric`),
  and the reflective optimizer defaults to it when no explicit `objectives` are
  given. Selection is now deterministic when latency is weighted out, so screening
  fitness and frontier selection agree on which axes matter. An explicit
  `objectives=` argument still overrides; default weights keep all four axes.

### Notes

- 1039 tests passing offline; ruff + mypy clean. The single known limitation
  documented in the 1.7.0 release (the reflective-optimizer flake) is now closed
  at its root cause rather than worked around.

See the [roadmap](ROADMAP.md) (1.7 milestone).

## [1.7.0] - 2026-06-17

Makes the spine honest and fast, and lays the model-registry foundation. The
advertised `Budget` becomes a hard cap, the 1.5 embeddings are wired into the
compiler so selection is semantic instead of bag-of-words, the streaming and
non-streaming run paths are unified, persistence moves off the event loop,
local-image input is fixed, and a data-driven `ModelRegistry` finally consumes
the underused `ModelProfile`. Every change is additive behind a new entry point
or opt-in flag on the frozen 1.0 API, all `@experimental`; promotions are now
gated on statistical significance instead of a point estimate.

### Added

- **Enforced full Budget on the single-shot run path.** `max_cost_usd` /
  `max_input_tokens` / `max_output_tokens` / `max_steps` are now hard caps on
  `app.run()` / `arun()`: a `BudgetUsage` is threaded through the model+tool loop
  and `exceeds()` is checked after each model call and tool round, raising the
  (previously dead) `BudgetExceededError` at the same choke point as residency
  and the cost SLO — recorded on the audit chain (`budget` decision) and the
  `budget.exceeded` event. A pre-flight input-token estimate is checked before
  the first call, and `BudgetAllocator` can reserve response + tool-loop tokens
  so it accounts for the full window. `RunConfig(enforce_budget_caps=False)`
  preserves the pre-1.7 soft-cap behavior for one minor.
- **Data-driven `ModelRegistry`** (`vincio.providers.registry`, exported as
  `ModelRegistry` / `default_model_registry`). A versioned, hot-reloadable,
  config-overridable catalog keyed by exact model id, instantiating
  `core.types.ModelProfile` (now carrying batch/cache pricing tiers, modalities,
  and GA/deprecation/retirement lifecycle dates). `ModelProvider.capabilities()`
  and `observability.costs.PriceTable` derive from it, with substring sniffing
  demoted to a last-resort fallback; an unknown-model lookup warns
  (`ModelUnknownWarning`) and emits `model.unknown` instead of silently billing
  $0. `importlib.metadata` entry-point groups (`vincio.providers` /
  `vincio.embedders` / `vincio.stores`) let third parties ship auto-registering
  adapters, and provider-native exact token counters register behind the
  `TokenCounter` Protocol (`register_token_counter`). Overlay a catalog with
  `VINCIO_MODEL_REGISTRY=<path.json|yaml>`.
- **Opt-in semantic context scoring** (`app.use_semantic_context_scoring()` /
  `retrieval.semantic_context_scoring`). When a real embedder is configured,
  context relevance, novelty, dedup, and conflict use cosine over the cached
  embeddings, the reranker's `upstream_relevance` is blended into relevance (no
  longer just a gate), and `_select` runs embedding-cosine maximal-marginal
  relevance with an `mmr_lambda` trade-off. The default stays lexical.
- **Value-level contradiction.** The compiler's negation-XOR conflict trigger is
  replaced by a salient-unit value-disagreement check: same-topic evidence that
  cites different numbers/dates (or flips polarity) is emitted as a structured
  conflict delta in the packet.
- **`RunHandle` + cooperative cancellation.** `app.submit(...)` returns a
  `RunHandle` whose `cancel()` propagates a cancellation into the run's
  bounded-concurrency groups; the cancelled run is still fully recorded on its
  trace and audit chain (a `CANCELLED` epilogue both run paths share). The
  streaming path gained the same `asyncio.timeout` latency deadline as the
  non-streaming path.
- **Async store contract.** `storage.base.asave` / `aquery` run a store's
  `save`/`query` off the event loop (`to_thread` for sync stores, native
  `asave`/`aquery` when present); the runtime now persists packets and runs
  without blocking the pipeline.
- **Significance-gated promotion.** `evals.experiments.ab_test` now returns a
  confidence interval and Cohen's-d effect size alongside the p-value, and the
  shared `evolution_loop` calls the t-test at the gate: a statistically
  significant regression on the primary metric blocks promotion, and an
  under-powered or non-significant gain is warned. The `loop_promotion` audit
  record carries the verdict.
- **Trace-replay executor.** `evals.replay.ReplayRunner(app).replay(traces,
  pin_tools=...)` re-runs captured trace inputs through a target app and diffs
  outputs, trajectory (`trace_diff`), and cost (`EvalReport.diff`), optionally
  pinning recorded tool outputs for determinism. Surfaced as `vincio trace
  replay --against <app>`.
- **Pluggable detector backends.** `security.DetectorBackend` / `DetectorSpan`
  let an ML model merge with the deterministic PII / injection / secret
  detectors; passing none keeps detection byte-for-byte unchanged.
- A new runnable example, `examples/31_honest_fast_spine.py`, and VincioBench
  metrics + CI budgets/SLOs across the **cost**, **rag**, **reliability**,
  **perf**, and **loop** families (budget-cap enforcement, unknown-model
  warning, embedding-MMR + value-contradiction, stream/non-stream parity,
  cancellation recording, inverted-index BM25, token memoization, registry
  lookup, significance-gated promotion, and replay fidelity).

### Changed

- **OpenAI local-image input is fixed.** A local image path is base64-encoded
  into a `data:` URL instead of an unreachable `file://` URL, via one shared
  `vincio.core.media` helper (with a byte-size cap) reused by the OpenAI,
  Anthropic, and Google chat providers and the multimodal embedders. Google
  also accepts Google-hosted image URIs (GCS `gs://` or the Files API host) via
  Gemini `fileData` (arbitrary public URLs, which Gemini cannot fetch, are not
  sent — supply a local path to inline them).
- **Truthful protocol capabilities.** A2A agent cards default
  `capabilities.streaming=False` until `message/stream` is actually dispatched;
  the MCP client's task-poll busy-loop is replaced with exponential backoff and a
  wall-clock deadline; and the A2A client polls `submitted`/`working` tasks to a
  terminal state instead of mis-reporting them as failed.
- **Hardened injection defense.** The injection detector runs a normalization +
  decode pre-pass (NFKC fold, zero-width strip, homoglyph/leetspeak fold,
  recursive base64/hex/rot13 decode, depth- and size-bounded) before its regex
  and heuristic signals, catching obfuscated attacks with no new false positives.
- **Tenant isolation can fail closed.** `AccessController(require_explicit_tenant
  =True)` stops treating an untagged (`tenant_id=None`) resource as globally
  readable — closing a cross-tenant fail-open. Defaults to the legacy behavior
  for one minor.
- **Evidence-gated compliance.** `ComplianceMapper` reads a control as `covered`
  only when backed by measured red-team / eval evidence; a configured-but-
  unmeasured control is now `partial` (structural, by-construction guarantees
  stay `covered`).
- **Sub-quadratic hot paths.** BM25 search scans inverted posting lists instead
  of every document per query term, `_select` selects incrementally (O(n·k))
  with inverted-index blocking for dedup/conflict, `count_tokens` is memoized,
  and the local vector index gains an optional numpy path — all behind
  availability checks, pure-Python staying the zero-dependency default.
- **1034 tests passing offline in ~4.5s; ruff + mypy clean**; thirty-one runnable
  examples; the VincioBench `cost` / `rag` / `reliability` / `perf` / `loop`
  families hold the 1.7 guarantees under CI-gated budgets.

## [1.6.1] - 2026-06-16

Completes the 1.6 governance follow-ups (no gaps): a real type-check gate, a
stronger residency control, and signable content credentials. Additive and
backward-compatible.

### Added

- **mypy is now a CI gate.** A new `Types (mypy)` job runs `mypy vincio` on every
  PR; the whole package type-checks clean (0 errors across 230 modules). Fixing
  the type errors this surfaced also hardened several latent issues — a
  mislabeled `HealthAwareFailover._ordered` return type, a `StateGraph` frontier
  dedup that used `set.add` as a value, an unguarded `anomaly_factor` multiply,
  and tightened `evidence_ids` / event-handler / finish-reason typing.
- **Residency endpoint-region inference.** `ResidencyPolicy` now infers the
  provider region from a region-bearing endpoint URL (AWS `us-east-1`-style,
  GCP/Vertex `europe-west4`-style, and sovereign-gateway jurisdiction
  subdomains) via the new `infer_region_from_url`, and run egress checks read the
  configured `provider.base_urls`. Matching is jurisdiction-aware:
  `allowed_regions=["eu"]` admits `eu-west-1` and `europe-west4`. Combined with a
  region-pinned endpoint, the egress-refusal control now reflects the real
  endpoint rather than only a hand-maintained map.
- **Signable synthetic-content manifests.** `mark_synthetic_content(...,
  signer=...)` attaches a cryptographic signature over a deterministic binding
  payload, and `verify_manifest(manifest, content, signer=...)` checks both the
  SHA-256 content binding and the signature (failing closed when a signature is
  present but no verifier is supplied). A dependency-free `HmacSigner`
  (HMAC-SHA256 over a `SecretString`) ships built in; supply your own
  `ContentSigner` for asymmetric, third-party-verifiable provenance.
  `app.content_signer` signs every auto-marked run.

### Changed

- `vincio.governance` gains `infer_region_from_url`, `ContentSigner`,
  `HmacSigner`, and `verify_manifest` (additive). `EvalRunner` and
  `gather_bounded` accept `Sequence` inputs; `PIIDetector(locales=...)` accepts
  any `Sequence`. `GovernanceConfig.card_format` is now a validated `Literal`.

### Quality

- **986 tests passing offline; ruff clean; mypy clean; VincioBench 131/131
  budgets** (two new governance budgets gate residency inference and signature
  verification); thirty runnable examples.

## [1.6.0] - 2026-06-16

Enterprise governance & compliance. Turns the audit and security spine into the
evidence regulated buyers require — model/system cards, OWASP/NIST/MITRE control
coverage, an AI-BOM, EU AI Act transparency artifacts, data lineage with
right-to-erasure, data-residency routing, multilingual PII, and RAG-poisoning
detection — all generated in the library from data Vincio already holds.
Additive behind `@experimental` 1.6 entry points on the frozen 1.0 API,
dependency-free; no public symbol removed or repurposed.

### Added

- **Model & system cards.** `vincio.governance.generate_model_card` /
  `generate_system_card`, `app.model_card()` / `app.system_card()`, and `vincio
  governance card` generate machine-readable cards from the live configuration
  and optional `EvalReport` evidence. A model card carries id/version,
  capabilities, limitations, and live pricing; a system card adds retrieval,
  memory, safety filters, human-oversight points, and governance controls. The
  schema is pluggable (`CardFormat`: Vincio native, Open Model Card, EU "AI
  Cards") and rendered from one captured fact set.
- **Compliance-framework mapping.** `ComplianceMapper` / `map_compliance` /
  `app.compliance_report()` / `vincio governance report` map a data-driven
  control catalog (`CONTROL_CATALOG`) for **OWASP LLM Top 10 (2025)**, **OWASP
  Agentic AI**, **NIST AI RMF (GenAI profile)**, and **MITRE ATLAS** onto
  Vincio's capabilities, backed by measured evidence — `RedTeamSuite` probe
  outcomes, the security configuration, and `EvalReport` metrics. The
  `ComplianceReport` is a `covered`/`partial`/`not_covered` matrix with the
  evidence string for each control, `coverage_rate`, `by_framework()`, `gaps()`,
  and `to_markdown()`.
- **AI-BOM.** `generate_aibom` / `app.aibom()` / `vincio governance aibom`
  produce a CycloneDX-1.6 AI bill of materials (base model + version,
  embedding/rerank models, fine-tune datasets, prompt/registry versions) as
  `machine-learning-model` / `data` components with optional **SHA-256 hashes**;
  `sha256_file` / `sha256_text`, `AIComponent.verify`, and `AIBOM.verify_all`
  support blast-radius assessment. Complements the shipped dependency SBOM + SLSA
  provenance.
- **EU AI Act transparency.** `mark_synthetic_content` emits a C2PA-style
  `ProvenanceManifest` (IPTC `trainedAlgorithmicMedia`, bound to the output by
  SHA-256), `ai_disclosure` returns a localized AI-interaction disclosure, and
  `data_summary` exports a grounding-data summary. `governance.content_marking`
  (or `app.content_marking`) attaches the manifest + disclosure to every run's
  `result.metadata`.
- **Data lineage & erasure-by-source.** A `LineageIndex` records source →
  document → chunk → evidence → output as the app ingests and runs
  (`app.trace_lineage(...)`); `app.erase_source(...)` / `vincio governance erase`
  satisfies a GDPR right-to-erasure across **every index, memory, and cache**,
  logged on the hash-chained audit chain (`erase_source`) and idempotent. Returns
  an `ErasureResult`.
- **Data-residency-aware routing.** `ResidencyPolicy` / `app.set_residency(...)`
  / `governance.allowed_regions` pin allowed provider regions and **refuse
  egress** to others as a blocking `PolicyViolation` recorded as a
  `residency_check` deny (raising `ResidencyViolationError`), enforced at the
  provider-resolution choke point before any request leaves the process.
- **Multilingual PII.** Non-English locale packs (`vincio.security.locales`:
  France, Germany, Spain, India, Singapore, Brazil, UK national-ID and phone
  formats) via `PIIDetector(locales=[...])`, `available_locales`,
  `get_locale_pack`, and `governance.locales` — layered on the English path
  without changing it (`PIIMatch.type` widened to `str` with a `locale` tag;
  built-in `PIIType` unchanged and still accepted).
- **Per-language eval slicing.** `EvalReport.slice`, `slice_by_tag`, and
  `tag_gap` surface the high-vs-low-resource accuracy gap so it can't hide in an
  aggregate.
- **Tokenizer fertility telemetry.** `FertilityTracker` / `app.fertility` track
  tokens-per-word/char per language and tenant, exposing the non-English "token
  tax" (`token_tax(language)`) so it is visible and routable; recorded
  automatically on each run from `UserInput.locale`.
- **RAG-poisoning detection.** `PoisoningDetector` / `PoisonVerdict` /
  `PoisoningReport` flag likely-poisoned retrieved evidence from
  authority/provenance signals (embedded instructions, low-authority/high-
  promotion sources, consensus outliers), with an optional async classifier hook
  and FP/FN telemetry (`PoisoningReport.telemetry`).
- **Config.** A new `governance` section (`GovernanceConfig`): `allowed_regions`,
  `provider_regions`, `deny_on_unknown_region`, `content_marking`, `locales`,
  `card_format`.
- **CLI.** `vincio governance card | report | aibom | lineage | erase`.
- **Errors.** `GovernanceError`, `ResidencyViolationError`, `ErasureError`.
- **Example & docs.** `examples/30_governance_compliance.py`; a new
  [governance guide](docs/guides/governance.md); API/CLI/config reference and
  SECURITY/ROADMAP updates.
- **VincioBench.** A new `governance` family gating card/AI-BOM completeness,
  framework-mapping coverage, erasure correctness, multilingual PII recall, and
  RAG-poisoning telemetry — 13 new `budgets.json` budgets (129 total) and three
  new SLOs.

### Changed

- `vincio.__all__` gains `ModelCard`, `SystemCard`, `ComplianceReport`,
  `ComplianceFramework`, `AIBOM`, `ResidencyPolicy`, `LineageRecord`,
  `ErasureResult`, `ProvenanceManifest`, `FertilityTracker`, and
  `PoisoningDetector` (additive; the frozen surface only grows).
- `PIIMatch.type` is now `str` (was a closed `Literal`) with a new optional
  `locale` field, so locale packs can contribute new category labels. Backward
  compatible — the built-in categories are unchanged and still accepted.

### Quality

- **980 tests passing offline; ruff clean; VincioBench 129/129 budgets**; thirty
  runnable examples.

## [1.5.0] - 2026-06-16

Multimodal, embeddings & retrieval breadth (vs LlamaIndex, Voyage/Cohere).
Keeps retrieval best-in-field as the embedding and ingestion frontier moves —
every new embedder, store, and parser sits behind an interface that already
existed. Additive under the frozen 1.0 API; no public symbol removed or
repurposed.

### Added

- **Matryoshka (MRL) embeddings.** `build_embedder(kind, dimensions=N)` (and the
  experimental `MatryoshkaEmbedder`, the `retrieval.embedding_dimensions` config
  field, and `mrl_truncate`) truncate each output vector to its `N` leading
  dimensions and L2-renormalize. Hosted embedders (Jina/Voyage/Cohere) request
  the shorter vector natively; everything else is wrapped, so the result is
  exactly `N` long. Storage/latency vs. recall is gated per dimension in the
  VincioBench `rag` family.
- **Query-vs-document input-type hints.** All built-in embedders accept an
  optional `input_type` (`"document"` / `"query"`); `VectorIndex` passes the
  right one on add vs. search. The `embed_texts(embedder, texts, input_type=...)`
  helper dispatches the hint only to embedders that support it, so custom
  embedders implementing only `embed(texts)` keep working unchanged.
- **Contextual & multimodal embedders.** `VoyageContextualEmbedder`
  (`voyage-context-3`, chunk vectors carry document context — complements
  `contextualize_chunks`) and unified text+image embedders
  `VoyageMultimodalEmbedder` (`voyage-multimodal-3`) and
  `CohereMultimodalEmbedder` (`embed-v4.0`) via `build_embedder`,
  `MultimodalInput`, and `embed_multimodal`. All ride core `httpx` — no SDK.
- **Five new vector stores.** Weaviate, Milvus, Elasticsearch/OpenSearch, and
  Vespa behind the one `build_vector_index` factory and the `Index` protocol,
  joining Qdrant, pgvector, Chroma, Pinecone, and LanceDB. Each lazy-imports its
  SDK with a helpful `StorageError` and accepts an injected client for offline
  round-trip tests. New extras: `vincio[weaviate|milvus|elasticsearch|opensearch|vespa]`.
- **Layout-aware PDF extraction.** `load_document(path, layout=True)` /
  `load_pdf(path, layout=True)` / `extract_pdf_layout` recover column-aware
  reading order, tables with bounding boxes, and figure regions for complex PDFs
  via `vincio[pdf-layout]` (pdfplumber); the dependency-free pypdf text path
  stays the default. Pure, offline-tested helpers `group_words_into_lines` /
  `order_blocks` / `assemble_layout`.
- **Voice / realtime (optional module).** `vincio.realtime`: a provider-neutral
  `RealtimeSession` over OpenAI Realtime / Gemini Live (WebSocket) or a
  deterministic in-process backend, with VAD, interruption (barge-in), and
  **in-session tool calls routed through the permissioned, sandboxed, audited
  tool runtime** (`app.realtime_session(...)`). A separate `vincio[realtime]`
  extra, `@experimental`, explicitly scoped as a stateful bidirectional module —
  not core context engineering.
- **New top-level symbols:** `MatryoshkaEmbedder`, `RealtimeSession`
  (both `@experimental`, since 1.5). Example `29_multimodal_retrieval.py`.

### Notes

- 919 tests passing offline; ruff clean; VincioBench 116/116 budgets;
  twenty-nine examples. The `rag` family gained MRL recall-vs-dimension and
  unified multimodal recall/MRR (four new budgets, three new SLOs).

See the [roadmap](ROADMAP.md) (1.5 milestone).

## [1.4.1] - 2026-06-16

Completes the 1.4 distillation-flywheel capture so faithful, grounded training
data needs no opt-in and covers every run path. Additive under the frozen 1.0
API; no public symbol removed or repurposed.

### Added

- **Flag-free faithful export from `RunResult`s.** A `RunResult` already carries
  the full untruncated output (`raw_text`) and the full cited evidence
  (`evidence` / `citations`), and the runtime now stamps the original input on
  `result.metadata["input"]` — so `app.export_training_set(runs=[...])` /
  `export_training_set_from_runs(...)` build grounding-checked, deduped,
  provenance-stamped fine-tuning JSONL **without `enable_training_capture()`**.
  The trace-based path stays for the "I only have traces" case.

### Fixed

- **Training capture now covers streaming runs.** `app.astream` records the full
  output and cited evidence on its trace when `training_capture` is on (and a
  truncated `output` span attribute for parity with non-streaming), so
  streaming-sourced traces curate into faithful training data too — previously
  only the `run` / `arun` / `batch` / eval path was instrumented.

### Notes

- 866 tests passing offline; ruff clean; VincioBench 112/112 budgets;
  twenty-eight examples. The two follow-ups documented in the 1.4.0 release are
  now closed: faithful capture no longer requires an opt-in flag (via the
  `RunResult` path), and streaming runs are covered.

See the [roadmap](ROADMAP.md) (1.4 milestone).

## [1.4.0] - 2026-06-15

Reflective optimization & the data flywheel (vs DSPy 3). 0.8 shipped the closed
loop; 1.4 sharpens the optimizer to the 2025–26 state of the art and adds the
lever the field is missing — turning production traces into cheaper inference —
while keeping every promotion gated, grounded, and audited. Like the rest of the
1.x line, the milestone is **additive under the frozen 1.0 API**: new surfaces sit
behind `@experimental` entry points, no public symbol is removed or repurposed,
and it uses only the core `httpx` dependency — no SDKs.

### Added

- **Reflective optimizer (GEPA-style)** (`vincio.optimize.ReflectiveOptimizer`,
  `ReflectiveResult`, `Reflector`, `HeuristicReflector`, `LLMReflector`,
  `MIPROProposer`, `ProposedEdit`, `Reflection`, `apply_edits`). Instead of blind
  mutation, the optimizer reads the eval report's failures, reflects on why a
  prompt lost, and proposes targeted edits, evolving a `ParetoFrontier`. A child
  is screened on a minibatch and earns a full rollout only when it beats its
  parent, so the GEPA sample-efficiency win holds under a **hard evaluation
  budget**, deterministic under seed. `strategy="mipro"` switches to MIPROv2-style
  joint instruction+example proposal. The result is a drop-in `OptimizationResult`:
  `ImprovementLoop(optimizer="reflective")`, `app.reflective_optimize(...)`, and
  `vincio optimize reflective` (and `vincio loop run --reflective`) promote through
  the identical gated path (registry push, eval-link, audit, event).
- **Distillation / fine-tune flywheel** (`vincio.optimize.export_training_set`,
  `TrainingSet`, `TrainingExample`, `BootstrapFinetune`, `DistillationResult`).
  `app.export_training_set(...)` / `vincio distill` curate production traces
  (feedback-filtered, grounding-checked against cited evidence, deduped, with full
  provenance) into provider-ready fine-tuning **JSONL** (OpenAI and Anthropic
  shapes); a teacher→student loop measures whether a cheaper student holds quality
  on the eval suite before promoting it into a runtime `ModelCascade`. Every
  exported example is grounded and gated. Opt-in `app.enable_training_capture()`
  (config `observability.training_capture`) records the full output and cited
  evidence on each trace so the export is faithful, not truncated to the span.
- **Learned prompt compression** (`vincio.context.LLMLinguaCompressor`,
  `TokenImportanceScorer`, `compression_faithfulness`, `faithfulness_preserved`,
  `salient_units`). A token-importance compressor that drops low-information tokens
  while protecting numbers, entities, citations, and query terms — a drop-in
  `ContextCompiler.compressor` alongside extractive compression.
  `vincio.optimize.CompressionTuner` / `app.gate_compression(...)` adopt it only
  when it preserves the cited-fact set and holds quality under eval;
  `app.use_learned_compression()` installs it directly.
- **Optimizer-judge calibration** (`vincio.optimize.JudgeCalibrator`,
  `JudgeStepReflector`, `JudgeStepProposal`, `JudgeCalibrationResult`).
  `app.calibrate_judge(...)` reflectively tunes a `GEvalJudge`'s evaluation steps
  against κ-validated human labels, adopting a new procedure only when its Cohen's
  κ strictly beats the incumbent, and leaving the judge's gating weight reflecting
  the higher agreement.
- New top-level exports: `ReflectiveOptimizer`, `TrainingSet`, `BootstrapFinetune`,
  `LLMLinguaCompressor`, `JudgeCalibrator`. New example
  `28_reflective_optimization.py`. The VincioBench `loop` family gains
  reflective-search-vs-baseline lift, distillation grounded-only export +
  quality-hold, and compression fidelity + faithfulness-gating gates (nine new
  budgets, three new SLOs).

### Notes

- **854 tests passing offline; ruff clean; VincioBench 112/112 budgets**;
  twenty-eight runnable examples. All 1.4 surfaces are `@experimental(since="1.4")`
  on the frozen 1.0 API — no existing behaviour changes, and the default compressor
  remains extractive until a learned one is installed.

See the [roadmap](ROADMAP.md) (1.4 milestone).

## [1.3.1] - 2026-06-15

Completes the 1.3 cost-and-reliability layer so it has no attribution or
behavioral gaps. All additive/fixes under the frozen 1.0 API.

### Added

- **Cost attribution now spans agents and crews.** `app.agent(...).run(...)` and
  `Crew.run` / `arun` accept `tenant_id` / `user_id` / `feature`, and every agent
  step and crew (manager + member) model call is recorded on the app's
  `CostLedger` — `app.cost_report` and budgets now cover agentic workloads, not
  just the `run` / `arun` / `astream` / `batch` pipeline.

### Fixed

- **Runtime cascades now escalate on streaming runs.** `app.astream` with a
  cascade buffers each rung and streams the accepted (escalated) answer, instead
  of silently using only the first rung — streaming and non-streaming runs now
  behave identically.
- **Response-cache hits are free.** A `response_cache` hit served the answer
  without an API call, so it is billed `$0` (and recorded as a `$0` cost event)
  rather than at the full uncached price; `cost_report` reflects real spend.

### Internal

- Hardened from an adversarial review: `RateLimiter.acquire` is lock-guarded;
  `KeyPool.stream` no longer falls back to a known-open breaker; the circuit
  breaker releases a half-open probe slot on a cancelled probe; self-correction
  cost is recorded on the ledger; `LiveIndex` keeps unchanged chunks' freshness
  consistent; Anthropic multi-part messages honor `cache_hint`. Strengthened
  offline batch-wire tests (OpenAI error files / failed status; Anthropic errored
  results / cancel). 797 tests; ruff clean; VincioBench 103/103.

See the [roadmap](ROADMAP.md) (1.3 milestone).

## [1.3.0] - 2026-06-15

Cost, reliability & scale (FinOps + resilience). What real teams hit when an LLM
app meets production traffic — provider outages, rate limits, runaway spend, and
the need to attribute every dollar — handled **in your application, not a proxy
hop**. Like 1.1/1.2, the milestone is **additive under the frozen 1.0 API**: new
surfaces sit behind `@experimental` entry points, no public symbol is removed or
repurposed, and it uses only the core `httpx` dependency — no SDKs.

### Added

- **Batch execution** (`vincio.providers.BatchRunner`, `BatchRequest`,
  `BatchResult`, `BatchJob`, `BatchRunResult`, `BatchBackend`,
  `InProcessBatchBackend`, `OpenAIBatchBackend`, `AnthropicBatchBackend`) —
  `app.batch([...])` / `app.abatch` / `vincio batch` submit a request set to the
  OpenAI **Batch API** or Anthropic **Message Batches API** (flat ~50% cost), poll
  to completion, and reconcile responses **by custom id** with partial-failure
  surfacing (missing ids become failed results, never dropped). The in-process
  backend is the offline/default path; the wire backends drive the real endpoints
  over the provider's own `httpx` client, reusing its payload-building and parsing.
  Same `RunResult` contract, cost-tracked at the discounted rate and traced.
- **Circuit breaking & health-aware failover** (`vincio.providers.CircuitBreaker`,
  `CircuitState`, `HealthAwareFailover`, `CircuitOpenError`) — a breaker tracks
  per-provider failure rate **and** latency over a rolling window, opens on
  threshold with half-open probing, and fast-fails (non-retryable) so the failover
  chain steers to healthy entries in microseconds. The documented pattern, made
  explicit: retries for transient (`RetryingProvider`), fallback for persistent
  (`HealthAwareFailover`), circuit-break for systemic (`CircuitBreaker`).
- **Key pooling & rate limiting** (`vincio.providers.KeyPool`, `RateLimiter`) —
  round-robins health-aware across multiple API keys/regions, enforces per-key
  dual **RPM + TPM** token buckets so a limit self-heals instead of erroring, and
  applies full-jitter backoff that honors `retry_after` on 429.
- **Runtime model cascades** (`vincio.optimize.ModelCascade`, `CascadeRung`,
  `response_confidence`) — `app.use_cascade([...])` starts on the cheapest rung and
  escalates only when a response's confidence is below the rung threshold (default:
  a clean, schema-valid stop is confident); a custom confidence callable drives it
  from your own metric. The offline `RoutingOptimizer` keeps tuning thresholds.
- **Cost attribution & budget SLOs** (`vincio.observability.finops`: `CostLedger`,
  `CostEvent`, `CostReport`, `CostBudget`, `BudgetManager`, `BudgetDecision`) —
  every model call in a `ContextApp` run (including its tool loop, self-correction,
  and batch) records an attributed `CostEvent` (`tenant` / `user` / `feature` /
  `run`), rolled up by any dimension (`app.cost_report(by=...)` /
  `vincio cost report --by tenant|feature`). `app.set_cost_budget(...)` enforces a
  per-scope budget on breach — **hard cap** (deny), **degrade-to-cheaper-model**,
  or **queue-to-batch** — as a `PolicyViolation` on the hash-chained audit path; an
  `anomaly_factor` raises a `cost.anomaly` event on a spend spike. Attribution is
  captured at request creation, so long agentic traces are counted honestly.
- **Provider-aware prompt caching** (`vincio.providers.PromptCacheStrategy`,
  `cache_hit_rate`) — `app.enable_prompt_caching(ttl="5m"|"1h")` attaches an
  Anthropic `cache_control` breakpoint with the chosen TTL to the compiler's stable
  prefix (when long enough to be worth caching); auto-cache providers (OpenAI/Gemini)
  rely on the stable→volatile ordering the compiler already produces. **Cache-hit
  rate** is recorded on every model span. On by default (`cache.provider_cache`).
- **Incremental & sharded indexing** (`vincio.retrieval.ShardedIndex`,
  `UpsertStats`) — `LiveIndex.upsert` gained **content-hash change detection** so
  only changed chunks re-embed, `LiveIndex.upsert_stream` for streaming ingestion,
  and `ShardedIndex` splits a corpus across N backends queried in parallel and
  merged, behind the existing `Index` protocol (a document's chunks co-locate).
- **VincioBench `scale` family** — gates batch-result correctness, circuit/failover
  recovery, prompt-cache hit rate, cost-attribution accuracy, and cascade savings;
  four new SLOs hold them (**103 budgets total, all green**).
- Example `27_cost_and_reliability.py` (27 examples, all run offline). New guide:
  [Cost, reliability & scale](docs/guides/cost-and-reliability.md); new comparison:
  [vs LiteLLM / gateways](docs/comparisons/litellm.md).

### Changed

- `__version__` is now `1.3.0`. `UserInput` gains an optional `feature` field and
  `Message` gains an optional `cache_ttl` field; `arun` / `astream` accept a
  `feature=` attribution argument; `CacheConfig` gains `provider_cache` /
  `provider_cache_ttl` / `provider_cache_min_prefix_tokens`. `HTTPProvider` gains
  `_get_json` / `_get_text` helpers and the Anthropic adapter sends the
  extended-cache-ttl beta header. All additive and backward-compatible.

See the [roadmap](ROADMAP.md) (1.3 milestone) for the full picture.

## [1.2.0] - 2026-06-14

Agentic evaluation & continuous quality. Vincio could run and trace a crew, a
graph, and a tool loop — 1.2 makes it **score** them: over the trajectory, over a
multi-turn conversation, and over live traffic. Every new metric is the same
object reused as an offline gate, a runtime guardrail, and an optimizer fitness
term. Like 1.1, the milestone is **additive under the frozen 1.0 API** — new
surfaces sit behind `@experimental` entry points, no public symbol is removed or
repurposed — and runs in your process with no hosted dependency.

### Added

- **Trajectory & tool-use metrics** (`vincio.evals.metrics`) — `tool_call_accuracy`
  / `tool_call_f1` (right tool, right args, in the right order),  `goal_accuracy`
  (successful termination + answer match), `plan_adherence` (LCS vs the expected
  plan), `plan_quality` (failed/redundant steps, reference-free),
  `step_efficiency` (steps vs an optimal path), and `topic_adherence`. They read a
  provider-neutral `Trajectory` (`vincio.evals.trajectory`) carried on the
  `RunOutput`, built with `RunOutput.from_agent_state(state)` /
  `from_crew_result(result)` / `from_trace(trace)` — a crew, a `StateGraph` run,
  or a captured trace is scored without re-instrumentation. Expected/optimal
  references live in `rubric['expected_tools' | 'plan' | 'optimal_steps' |
  'topic']`. `EvalReport.metric_families()` splits the report into final-output-only
  vs trajectory evaluation.
- **Conversational metrics** — `conversation_outcome` (did the thread achieve the
  user's goal) and `intent_resolution` (fraction of user turns addressed), joining
  `knowledge_retention` / `conversation_relevance`.
- **Multi-turn simulator** (`vincio.evals.Simulator`, `Persona`,
  `SimulatedConversation`, experimental) — drives multi-turn sessions from a
  persona + goal; LLM-backed with a seeded template fallback, so it is
  deterministic offline (same seed → identical conversation).
  `SimulatedConversation.to_eval_case()` feeds the conversational metrics;
  `dataset_from_traces(..., group_by_session=True)` stitches a session's traces
  into a multi-turn golden case.
- **Online / continuous eval** — `app.add_online_evaluator(metric,
  sample_rate=...)` (experimental) scores a sampled fraction of live runs after
  the response is finalized (scheduled off the hot path; `app.aflush_online()`
  drains in tests), writing each score as a time series on the metadata store
  (`OnlineEvaluator.series()`). No traffic mirrored to any external service.
- **Drift detection** (`vincio.evals.DriftMonitor`, `DriftReport`) — rolling
  score drift and embedding-distribution drift of inputs against the golden-set
  distribution; raises a `drift.detected` event on the bus and persists baselines
  (`drift_baselines`). `vincio eval drift baseline.json current.json` reports it.
- **Human-in-the-loop annotation** (`vincio.evals.AnnotationQueue`,
  `cohens_kappa`) — records human labels next to LLM-judge scores and tracks
  **Cohen's κ**; `GEvalJudge.calibrate()` now also returns `cohens_kappa`, and
  `judge.gating_weight(threshold)` / `queue.judge_trusted()` gate a judge on
  agreement. `vincio eval annotate labels.jsonl` reports it.
- **Production A/B** — `app.experiment(name, variants=..., dataset=...,
  metrics=...)` (experimental) returns an `Experiment` comparing variants on eval
  metrics **and** cost (`.compare()` / `.cost()` / `.significance(metric)`) with
  the paired/Welch tests `ExperimentTracker` already ships.
- **Metric-as-guardrail** — `app.add_metric_rail(metric, threshold=...)` /
  `vincio.evals.metric_guardrail(metric, threshold=...)` wrap any metric as a
  deterministic runtime rail predicate (direction from `LOWER_IS_BETTER`).
- **Optimizer interconnection** — `vincio.optimize.AGENTIC_OBJECTIVES`, a Pareto
  objective preset over `goal_accuracy` / `tool_call_accuracy` / `step_efficiency`
  / `cost`; trajectory metrics are ordinary metrics, so they flow into
  `report.metric_values` and the frontier unchanged.
- **VincioBench `agentic_evals` family** — gates trajectory-metric agreement
  against labeled traces, the output-only/trajectory gap, simulator determinism,
  drift sensitivity/specificity, and κ tracking; six new SLOs hold them (94
  budgets total, all green).
- Example `26_agentic_eval.py` and a labeled golden set
  `tests/golden/agentic_eval.jsonl` (26 examples, all run offline). New guide:
  [Agentic evaluation & continuous quality](docs/guides/agentic-eval.md).

### Fixed

- **Gemini embedding cost tracked as $0** — the cost table referenced the dead
  `text-embedding-004` while the Google provider defaults to `gemini-embedding-001`,
  which was absent from the table, so a price lookup fell through to the zero
  default and embedding cost was billed at $0. `gemini-embedding-001` is now priced
  ($0.15 / 1M input tokens), with a regression test.

### Changed

- `__version__` is now `1.2.0`. `RunOutput` gains an optional `trajectory` field
  and `from_agent_state` / `from_crew_result` / `from_trace` constructors;
  `dataset_from_traces` gains `group_by_session`; `GEvalJudge.calibrate` also
  returns `cohens_kappa`. All additive and backward-compatible.

See the [roadmap](ROADMAP.md) (1.2 milestone) for the full picture.

## [1.1.0] - 2026-06-13

Protocols & interoperability — the first post-1.0 milestone. Vincio now speaks
the interoperability protocols the ecosystem standardized on in 2025–26 —
**MCP** (client *and* server), **A2A** agent-to-agent, and Anthropic **Agent
Skills** — plus a unified reasoning control across providers. Everything is
**additive under the frozen 1.0 API**: every new surface sits behind a new
entry point and is marked `@experimental`; no public symbol is removed or
repurposed, so upgrading across the 1.x line never breaks working code. The new
protocols use only the core `httpx` dependency — no SDKs — and run in your
process; Vincio adopts the standards, it does not become a service.

### Added

- **MCP client + server** (`vincio.mcp`, experimental) — `MCPClient` /
  `app.add_mcp_server(name, command=/url=/server=)` connect to MCP servers over
  **stdio**, **Streamable HTTP**, and an **in-process** transport (offline
  tests), negotiate capabilities, and surface `tools` / `resources` / `prompts`.
  MCP tools register through the *existing* permissioned, sandboxed, audited,
  budgeted tool runtime (namespaced `<server>.<tool>`); MCP resources become
  evidence with `origin: mcp:<server>` provenance; MCP prompts import as
  `PromptSpec`. Server-initiated **sampling** routes to the app's provider,
  **elicitation** to a human-gate callback; OAuth 2.1 seams (`pkce_pair`,
  `static_token_validator`) and a long-running **Tasks** poll path are included.
  `app.serve_mcp()` / `vincio mcp serve` expose a `ContextApp` as an MCP server
  (tools/resources/prompts), with the policy engine and audit log enforced on
  every inbound call and OAuth 2.1 resource-server token validation.
  `vincio mcp tools` / `mcp add` inspect and wire servers from the CLI.
- **A2A (agent-to-agent)** (`vincio.a2a`, experimental) — `app.serve_a2a(crew |
  graph | None)` serves an **Agent Card** (`/.well-known/agent.json`) and a
  JSON-RPC **task lifecycle** (`submitted → working → input-required →
  completed/failed`); graph human-in-the-loop interrupts surface as
  `input-required` and resume by `taskId`. `A2AClient` / `connect_a2a` reach
  remote agents, and `RemoteA2AAgent` plugs a remote agent into a local crew as
  a **bounded, traced** delegate. Token validation + per-task audit (`a2a_serve`).
- **Agent Skills** (`vincio.skills`, experimental) — `app.add_skill(path)` loads
  Anthropic-style `SKILL.md` (YAML frontmatter + Markdown + optional bundled
  scripts) and injects it through the compiler with **progressive disclosure**:
  a one-line index is always available; a skill's full body enters the budget
  only when the task is relevant (scored and cited like any evidence). Bundled
  scripts run as sandboxed, permissioned tools (`register_scripts=True`).
- **Unified reasoning control** — `RunConfig(reasoning_effort="minimal"|"low"|
  "medium"|"high")` / `thinking_budget_tokens` map to OpenAI reasoning effort,
  Anthropic extended thinking (sampling left at default), and Gemini thinking
  budgets; providers without reasoning ignore them. The negotiated reasoning
  mode is recorded on the `prompt_render` span and `reasoning_tokens` on the
  `model_call` span. `ModelCapabilities.reasoning` declares support.
- **OpenAI Responses API adapter** (`OpenAIResponsesProvider`,
  `build_provider("openai_responses")`) — stateful `previous_response_id`,
  built-in tools, reasoning preserved across tool calls, behind the same
  `ModelProvider` interface; Chat Completions stays the portable default.
- **VincioBench `protocols` family** — gates MCP tool schema-fidelity + resource
  provenance, A2A delegation termination, and Agent-Skill progressive-disclosure
  budget savings; three new SLOs hold them (88 budgets total, all green).
- Four new examples: `22_mcp_tools_and_resources.py`, `23_a2a_delegation.py`,
  `24_agent_skills.py`, `25_reasoning_control.py` (25 examples, all run offline).
- New guides: [MCP](docs/guides/mcp.md), [A2A](docs/guides/a2a.md),
  [Agent Skills](docs/guides/agent-skills.md), and
  [reasoning control](docs/guides/reasoning.md).

### Fixed

- **Reasoning-token cost accounting** — Gemini reported thinking tokens
  (`thoughtsTokenCount`) as `reasoning_tokens` but excluded them from the
  billable output (`candidatesTokenCount`), so thinking was costed at $0. The
  Google adapter now folds thinking tokens into the billable output (they are
  billed at the output rate, matching `totalTokenCount`), while
  `reasoning_tokens` keeps the thinking subset for telemetry. OpenAI/Anthropic
  were already correct (reasoning is part of completion/output tokens).

### Changed

- `__version__` is now `1.1.0`. `ModelRequest` gains `reasoning_effort`,
  `thinking_budget_tokens`, and `previous_response_id`; `RunConfig` gains
  `reasoning_effort` / `thinking_budget_tokens`; `ModelCapabilities` gains
  `reasoning`; `MockProvider(reasoning=True)` emulates thinking tokens offline.
  All additive and backward-compatible.

See the [roadmap](ROADMAP.md) (1.1 milestone) for the full picture.

## [1.0.0] - 2026-06-13

Stabilization & guarantees — the 1.0 roadmap milestone. This release does not
add subsystems; it turns the library into a product you can trust in
production. Every guarantee is mechanical: SemVer on a frozen public surface
with an enforceable deprecation policy, published SLOs that CI budgets hold at
least as strict, a documented threat model backed by offline audit-chain
verification and a resource-limited tool sandbox, supply-chain attestations on
releases, and a docs-completeness gate that runs every example.

### Added

- **API stability module** (`vincio.stability`) — `deprecated(since=,
  removed_in=, alternative=)` and `experimental(since=, note=)` decorators
  (working on functions and classes) that emit `VincioDeprecationWarning` /
  `VincioExperimentalWarning`; `deprecated_alias(...)` for renamed symbols;
  `stability_of(obj)` to introspect any symbol's contract; `public_api()` and
  `API_VERSION`. All are re-exported from the top-level `vincio` package, which
  is now the SemVer-covered public surface. See `docs/reference/stability.md`.
- **Published SLOs** (`benchmarks/slos.json`, `docs/reference/slo.md`) —
  latency/throughput/token-efficiency/quality/security targets, each naming the
  VincioBench budget that enforces it. The budget is held at least as strict as
  the public promise, so a passing CI run provably honors the SLO;
  `tests/test_slos.py` verifies the invariant.
- **Offline audit-chain verification** — `verify_audit_file(path)` and
  `AuditLog.verify_file()` re-read the persisted JSONL and validate the SHA-256
  hash chain, detecting tampering after a process restart and pinpointing the
  first broken line (`ChainVerification`). New CLI: `vincio audit verify [path]`.
- **Threat model** (`docs/security/threat-model.md`) — STRIDE over the real
  controls (access, audit, injection, PII/secrets, sandbox), with the explicit
  out-of-scope statement and the supply-chain story.
- **Supply-chain attestations** — the release workflow now generates a
  **CycloneDX SBOM** and emits **SLSA build-provenance attestations**
  (`actions/attest-build-provenance`) for the published wheel and sdist.
- **VincioBench methodology** (`benchmarks/METHODOLOGY.md`) — what each family
  measures, its naive baseline, corpus provenance, the budgets-vs-SLOs design,
  and how to reproduce every number offline. Reports now include an
  `environment` block (Vincio/Python versions, platform, schema version).
- **Security & governance example** (`examples/21_security_governance.py`) —
  PII/secret redaction, injection defense, RBAC/ABAC + tenant isolation,
  programmable rails, and a tamper-evident audit log, all offline.
- **Docs-completeness gate** (`tests/test_docs_completeness.py`,
  `tests/test_examples.py`) — runs all 22 examples end-to-end offline and
  asserts every public subsystem is documented and every example is indexed.
  The API reference now documents `vincio.input`, `vincio.documents`,
  `vincio.cli`, and `vincio.stability`.

### Changed

- **Tool sandbox hardening** — `run_subprocess_sandboxed` and `SandboxedPython`
  accept `max_cpu_seconds` / `max_memory_bytes` / `max_open_files` and apply
  them via POSIX `setrlimit` in the child (best-effort; the wall-clock timeout
  and output caps always apply). `SandboxedPython` defaults to conservative
  10s CPU / 512 MB / 64-fd limits.
- `__version__` is now `1.0.0`; the package classifier moves to
  `Development Status :: 5 - Production/Stable`. Top-level exports add
  `API_VERSION`, `StabilityLevel`, `VincioDeprecationWarning`,
  `VincioExperimentalWarning`, `deprecated`, `experimental`, and `stability_of`.
- `SECURITY.md` now lists 1.0.x as supported and documents SBOM/provenance.

### Fixed

- Carried forward from 0.9.0 and noted here for the 1.0 record: the
  `ContextApp.add_evaluator` key mismatch for nameless callables (e.g.
  `functools.partial`) — the name is resolved once so later metric lookup
  succeeds.

## [0.9.0] - 2026-06-13

Integrations, connectors & developer experience — the 0.9 roadmap milestone.
Win on coverage and ergonomics so real projects adopt Vincio without rewriting
their stack: an OpenAI-compatible passthrough for any endpoint, hosted
rerankers/embedders and three more vector stores behind the existing
interfaces, two-way LangChain/LlamaIndex interop, scaffolding templates with a
typed config schema, notebook reprs and an interactive TUI, opt-in domain
packs, and migration guides. Every new adapter implements an interface the
engine already speaks, so breadth adds no new concepts — context compilation,
budgeting, evals, traces, and security apply unchanged.

### Added

- **OpenAI-compatible passthrough** (`vincio.providers.openai_compat`) —
  `OpenAICompatibleProvider` reaches any Chat-Completions endpoint;
  `openai_compatible("groq")` / `openai_compatible(base_url=..., api_key=...)`
  construct one, with named presets for `groq`, `together`, `fireworks`,
  `openrouter`, `deepseek`, `perplexity`, `xai`, and `nvidia`. Presets are
  registered in the provider registry (so `build_provider("groq")` and
  `provider.default: groq` work) and their keys resolve from the conventional
  `<NAME>_API_KEY` env var — no extra wiring.
- **Hosted rerankers** (`vincio.retrieval.rerankers`) — `CohereReranker`,
  `JinaReranker`, and `VoyageReranker` call the real rerank endpoints over the
  core `httpx` dependency (no SDK), behind `build_reranker("cohere"|"jina"|
  "voyage", api_key=..., model=...)` and the `retrieval.reranker` config. An
  injectable `httpx.AsyncClient` keeps them offline-testable.
- **Hosted embedders** (`vincio.retrieval.embeddings`) — `JinaEmbedder`,
  `VoyageEmbedder`, and `CohereEmbedder` (Cohere's v2 `embeddings.float` shape
  handled), plus a `build_embedder("local"|"jina"|"voyage"|"cohere"|<provider>)`
  factory that also wraps any embedding-capable provider as a `ProviderEmbedder`.
- **More vector stores** — `ChromaVectorIndex`, `PineconeVectorIndex`, and
  `LanceDBVectorIndex` join Qdrant and pgvector behind the retrieval `Index`
  protocol, unified by `vincio.storage.build_vector_index(kind, embedder,
  **opts)` (`memory`, `qdrant`, `pgvector`, `chroma`, `pinecone`, `lancedb`).
  Missing optional dependencies raise a clear, actionable `StorageError`. New
  extras: `vincio[chroma]`, `vincio[pinecone]`, `vincio[lancedb]`.
- **Framework interop** (`vincio.interop`) — bring LangChain and LlamaIndex
  **tools, retrievers, loaders/readers, and embeddings** into Vincio, and hand
  Vincio's back. The `from_*` adapters are duck-typed (they import nothing
  heavy), so existing assets drop in without a new dependency;
  `add_langchain_tool` / `add_llamaindex_tool` register *and* enable a tool in
  one call; imported documents chunk, index, budget, and cite like any local
  file, and imported tools run through the same permissioned, sandboxed, audited
  runtime. The `to_*` adapters build real framework objects (extras
  `vincio[langchain]` / `vincio[llamaindex]`).
- **Scaffolding & templates** — `vincio init --template {minimal,rag,agent,eval}`
  generates a tailored `ContextApp`, `vincio.yaml`, golden set, and (for `rag`)
  sample docs. Every generated config carries a `# yaml-language-server:
  $schema=…` hint and ships a JSON Schema for editor completion; `--provider`
  sets the default provider.
- **Typed config tooling** — `config_json_schema()` derives a JSON Schema from
  the typed `VincioConfig`; `vincio config schema` emits it, `vincio config
  validate` checks a config file with clear errors, and `vincio config show`
  prints the effective merged configuration.
- **Notebook & TUI ergonomics** — `vincio.notebook.enable_rich_reprs()` attaches
  HTML/Markdown reprs to `RunResult`, `Trace`, `EvalReport`, `MemoryItem`, and
  `SearchHit` for Jupyter (pure `*_html`/`*_markdown` render functions you can
  also call directly; `enable_rich_reprs` is exported from the top level).
  `vincio.tui.TUI` / `vincio tui` is a dependency-free, keyboard-driven inspector
  for runs, traces, and memory, with pure screen renderers and injectable IO so
  it is fully unit-tested.
- **Domain packs** (`vincio.packs`) — opt-in, dependency-free bundles for
  **support, engineering, finance, and legal**: a role/objective/rules prompt
  config, a structured output schema, recommended policies + evaluators, and a
  golden eval set. `app.use_pack("support")` applies one through the public app
  API (layer your own settings on top); `load_pack` / `available_packs` /
  `register_pack` and `vincio packs list` / `vincio packs show` round it out.
- **Migration guides** — "coming from LangChain / LlamaIndex / Ragas / Mem0"
  guides that map concepts one-to-one, plus an
  [integrations guide](docs/guides/integrations.md) covering the new providers,
  vector stores, embedders, rerankers, and interop adapters. Two new runnable
  examples: `19_framework_interop.py` and `20_domain_pack.py`.

### Fixed

- `ContextApp.add_evaluator` registered a callable without `__name__` (e.g. a
  `functools.partial`) under a key one greater than the one it recorded in
  `app.evaluators`, so later lookup missed the metric; the name is now resolved
  once.
- Removed a duplicate `dist/` entry in `.gitignore`. (The provider-transport
  reliability fixes — event-loop-safe HTTP clients and 429 cooldowns honored
  from provider error bodies — shipped with 0.7/0.8 and are documented under
  [0.8.0].)

### Changed

- `__version__` is now `0.9.0`; top-level exports add `Pack`, `load_pack`,
  `available_packs`, and `enable_rich_reprs`. New offline tests cover provider
  presets and key resolution, hosted reranker/embedder wire formats, the
  vector-store factory, both interop bridges, pack loading/application/run, the
  notebook reprs, the TUI loop, and every new CLI command; the suite stays
  fully offline and ruff-clean.

## [0.8.0] - 2026-06-13

The closed-loop ecosystem — the 0.8 roadmap milestone, and the differentiator:
the milestone no single-purpose library can ship, because it requires owning
the whole lifecycle. One continuous, reproducible improvement cycle —
trace → dataset → eval → optimize → promote — plus the feedback paths that
let every organ tune the others: runs write grounded facts back to memory,
eval-scored relevance tunes retrieval, the optimizer keeps a cost/quality
Pareto frontier instead of one score, budget allocation is learned from eval
outcomes, and guided offline search strategies drive the evolution loop.

### Added

- **The improvement loop** (`vincio.optimize.loop`) — `ImprovementLoop` /
  `app.improvement_loop()` / `vincio loop run` wires the pieces that already
  exist into one call: capture the traces production runs already write
  (any exporter), curate them with `dataset_from_traces` (only successful
  runs whose mean user feedback clears `min_feedback_score`; the dataset's
  case-id fingerprint is recorded for reproducibility), evaluate the current
  prompt as the baseline, run the gated prompt optimizer, and promote the
  winner: pushed to the `PromptRegistry`, tagged (`production` by default),
  linked to the eval report that justified it, applied to the live app,
  written to the hash-chained audit log (`loop_promotion`), and announced on
  the event bus (`loop.promoted`). Baseline and winner reports land in the
  `ExperimentTracker` (same metadata store as runs), so `compare()` and
  `ab_test()` work across cycles; `dry_run=True` reports the decision
  without acting. Candidate evaluations are memory-write-free: an eval run
  never pollutes user memory or hands later candidates different recall
  state than earlier ones saw.
- **Auto-memory from runs** (`vincio.memory.facts`) — with
  `memory.write_back: [facts]`, verifiable claims from a run's output that
  the cited evidence supports become *candidate* memories:
  `extract_grounded_facts()` is deterministic (claim-shaped sentences,
  support-thresholded lexical grounding against the cited evidence,
  citation markers stripped), `MemoryEngine.write_back(facts=...)` writes
  them with measured support and evidence provenance
  (`origin: run_fact`, confidence scaling with support), and admission
  still runs the guarded write policy — privacy, stability, contradiction,
  confidence — with the candidate status penalty in recall until confirmed.
  New config: `memory.fact_min_support`, `memory.max_facts_per_run`.
- **Retrieval feedback** (`vincio.optimize.retrieval_feedback`) —
  `RetrievalFeedback` tunes a live `RetrievalEngine` from relevance labels
  that already live on eval cases (`rubric.relevant_ids`, via
  `records_from_dataset` / `records_from_report`): a deterministic
  coordinate search over per-index RRF fusion weights and a grid over the
  heuristic reranker's blend, both **gated** — weights change only when
  recall@k + MRR over the records measurably improve, and the engine is
  restored untouched otherwise. `recommend_chunking(reports_by_config)`
  picks the chunking config whose eval report scored best, staying on the
  baseline unless beaten by `min_improvement`.
- **Cost/quality Pareto optimization** (`vincio.optimize.pareto`) —
  `pareto_loop` keeps the full multi-objective frontier instead of one
  scalar: `ObjectiveSpec` axes (defaults: accuracy, groundedness, cost,
  latency), `ParetoFrontier` with non-dominated filtering, `knee()`
  (best summed normalized goodness), and `select(constraints=, prefer=)`
  for per-objective bounds like `{"cost": 0.01}`. Screening still uses
  scalar fitness (cheap); the final pick comes from the frontier of
  full-dataset reports and passes the same promotion safety rules as the
  scalar loop.
- **Learned context budgeting** (`vincio.optimize.budget_learning`) —
  `BudgetLearner` searches bounded perturbations of the per-task allocation
  tables (move a slice of budget between blocks, renormalize) and adopts a
  learned table only through gated promotion; `LearnedAllocations`
  persists as JSON and installs via `app.use_learned_budgets()` or
  `BudgetAllocator(learned=...)` — tasks without a learned table keep the
  fixed defaults.
- **Guided offline search strategies** (`vincio.optimize.strategies`) —
  `hill_climb` (single-knob mutations of the incumbent) and `anneal`
  (Metropolis acceptance with a cooling schedule) condition each proposal
  batch on subset scores already observed; both are deterministic under a
  seed, hard-bounded by the evaluation budget, and pluggable into
  `ContextOptimizer(strategy=...)` or usable directly via
  `guided_search()`. Pre-scored candidates flow into `evolution_loop`
  without re-screening, and `OptimizationResult` now carries the evaluated
  baseline candidate.
- **CLI** — `vincio loop run --app app.py [--dataset ds.jsonl |
  --min-feedback X] [--gate "metric=>= 0.9"] [--tag production]
  [--experiment NAME] [--dry-run]`.
- **Docs & examples** — a "close the loop" guide
  (`docs/guides/close-the-loop.md`), updated API/CLI/config references,
  0.8 sections in the DSPy and Ragas comparisons, and runnable example
  `18_closed_loop.py` (the full cycle offline: auto-memory, promotion,
  the frontier behind the decision, retrieval feedback, learned budgets).
- **VincioBench `loop` family** — promotion fires and is deterministic,
  gates block regressions, the registry version is tagged and eval-linked,
  grounded facts are written (and ungrounded ones never are), retrieval
  tuning improves and is gated, the frontier excludes dominated points with
  a balanced knee, learned budgets promote, and guided search respects its
  budget — under 14 new CI-gated budgets (81 total).

### Changed

- `OptimizationResult` gains a `baseline: Candidate` field (the evaluated
  baseline with its full report), so loop callers can log and compare it.
- `evolution_loop` skips subset screening for candidates that arrive with
  `subset_fitness` already set (guided-search support); fresh candidates
  behave exactly as before.
- `BudgetAllocator` accepts `learned=` per-task allocation tables that
  override the fixed `TASK_ALLOCATIONS` entry for their task type.
- `MemoryEngine.write_back` accepts `facts=` (a list of `GroundedFact`)
  alongside `evidence=` and `tool_results=`.
- **495 tests passing offline in ~2s; ruff clean**; eighteen runnable
  examples; 81 CI-gated VincioBench budgets.

## [0.7.0] - 2026-06-13

Structured output, guardrails & reliability — the 0.7 roadmap milestone.
Reliability as a guarantee, not a hope: provider-native constrained decoding
with strict schema sanitization, streaming validation with early abort,
DSPy-style typed signatures that feed the optimizer, programmable rails in
the deterministic policy engine, bounded self-correcting loops that never
invent facts, and multi-schema routing — every failure, repair, and rail
decision landing on the trace and in the hash-chained audit log.

### Added

- **Constrained generation** (`vincio.output.constrained`) —
  `to_strict_json_schema()` transforms any JSON schema for strict
  provider-native constrained decoding (every object closed via
  `additionalProperties: false`, every property required, optional fields
  made nullable, `default`/`format` stripped) while validation keeps running
  against the original schema; `negotiate_decoding()` picks
  `native`/`prompt`/`none` from the provider capability matrix per run, and
  the chosen mode is recorded on the `prompt_render` and `output_validation`
  spans. Grammar-style constraints `choice_schema(options)` and
  `regex_schema(pattern)` express fixed choices and regex-shaped strings as
  schemas that ride the same native path; the deterministic JSON-schema
  validator now also enforces `pattern`.
- **Streaming validation** (`vincio.output.streaming`) —
  `StreamingValidator` accumulates text deltas, parses the balanced partial
  JSON, and prefix-checks it against the schema (`validate_partial`):
  missing required fields are tolerated while streaming, definite
  mismatches — wrong type, unknown field on a closed object — are reported
  mid-stream. `app.astream()` wires it in automatically: `partial_output`
  events now carry `valid_prefix` and `validation_errors`, so consumers can
  abort a generation that can no longer be valid; `finalize()` applies the
  allowed structural repairs at stream end.
- **Typed signatures** (`vincio.prompts.signatures`) — DSPy-style
  input→output signatures over the prompt AST: subclass `Signature` with
  `InputField` / `OutputField` markers (docstring becomes the instruction)
  or use the string form
  `signature("question, context -> answer, confidence: float")`.
  `Signature.to_prompt_spec()` compiles to a `PromptSpec` (drop-in target
  for `PromptOptimizer` variants/rewrites); `Predict` /
  `app.predictor(sig)` executes with provider-native constrained decoding
  and the full validation pipeline, returning typed results
  (`result.label`, `result.confidence`); inputs are type-checked before the
  call.
- **Rails as policies** (`vincio.security.rails`) — programmable rails as
  plain data (`Rail`: kind `topic` / `format` / `safety` / `custom`,
  direction, action `block` / `warn` / `redact`, parameters) evaluated by
  `RailEngine` inside the deterministic policy engine: topic rails match
  blocked/allowed topics by word-boundary patterns, format rails check
  length and require/forbid regexes, safety rails reuse the security
  engine's PII detector, secret scanner, and injection detector
  (`action="redact"` masks PII instead of blocking), and custom rails call
  predicates registered via `app.register_rail_predicate()`. Input rails
  run before the model is called (a blocking violation denies the run);
  output rails run inside the validation pipeline's policy step. Every
  violation is a `PolicyViolation` named `rail:<name>` on the trace and in
  the audit log. New app APIs: `app.add_rail(...)`.
- **Self-correcting loops** (`vincio.output.correction`) — `SelfCorrector`
  runs bounded validate → critique → repair cycles: the critique is built
  deterministically from the `ValidationReport` (`build_critique`), the
  repair request is structure-only (re-serialize, rename, retype — never
  add, remove, or change factual content), semantic/citation/policy
  validators re-run every cycle, and the loop stops at the first valid
  output, `max_cycles`, or the hard `max_cost_usd` ceiling.
  `app.enable_self_correction(max_cycles=, max_cost_usd=)` wires it into
  the run flow; cycles, cost, and outcome are a `self_correction` trace
  event and audit-log details.
- **Multi-schema routing** (`vincio.output.routing`) — `SchemaRouter` holds
  named `SchemaRoute`s (schema + task types / keywords / predicate /
  priority): `route()` picks the output contract for a run before
  generation (keywords match at word starts, so "crash" matches
  "crashed"), `classify()` finds which registered schema some structured
  data matches, and `validate_any()` validates against the alternatives.
  `app.add_output_schema(schema, keywords=..., task_types=..., when=...)`
  routes per run; the chosen schema is recorded on the `prompt_render`
  span.
- **Interconnection** — every validation failure and repair is now a trace
  event (`repair` / `validation_failed` / `self_correction` /
  `stream_invalid_prefix` events on the `output_validation` and
  `model_call` spans) *and* an `output_validation` entry in the
  hash-chained audit log (`decision=repair|deny`, with errors, repairs, and
  correction cycles); rails reuse the security detectors; signatures feed
  the optimizer.
- **VincioBench** — new `reliability` family measures strict-schema closure
  (100% objects closed and fully required), mid-stream invalid detection
  with abort savings (~98% of an invalid output's tokens saved offline),
  self-correction recovery rate with cycle bounds, rail catch rate with
  zero false positives on clean text, signature prediction validity and
  optimizer variant generation, and schema-routing/classification accuracy
  — held by 13 new `budgets.json` gates in CI.
- **Docs & examples** — a new how-to guide
  (`docs/guides/reliability-guardrails.md`), an expanded structured-output
  guide (constrained decoding, streaming validation, routing, signatures,
  self-correction), comparison write-ups for Pydantic AI, Guardrails AI,
  and NeMo Guardrails, a typed-signatures section in the DSPy comparison,
  and runnable example `17_reliable_structured_output.py`; the examples
  index now also lists the 0.6 crew and durable-graph examples.

### Fixed

- **HTTP provider clients no longer die with the event loop** — a provider
  reused across `asyncio.run()` calls (the natural sync usage of
  `generate_sync` / `stream_sync` / `app.run`) recreates its pooled
  `httpx.AsyncClient` when the cached client is bound to a closed or
  different loop, instead of raising "Event loop is closed".
- **Rate-limit cooldowns are honored from error bodies** — when a 429
  carries no `Retry-After` header, the retry delay is extracted from the
  provider error body (Google's `RetryInfo.retryDelay` detail or
  "retry in Ns" message), and `RetryingProvider`'s backoff cap was raised
  from 20s to 60s, so free-tier per-minute limits self-heal inside the
  retry loop.
- **Gemini defaults match the live API** — the default Google embedding
  model is now `gemini-embedding-001` (the live batch-embedding model), and
  the price table covers the current GA models (`gemini-2.5-pro`,
  `gemini-2.5-flash`, `gemini-2.5-flash-lite`, `gemini-2.0-flash`,
  `gemini-2.0-flash-lite`, `text-embedding-004`) so cost tracking reports
  paid-tier rates instead of $0 for unknown models.

### Changed

- `RunStreamEvent` gains `valid_prefix` and `validation_errors` on
  `partial_output` events (streaming validation).
- `PolicyEngine` accepts a `rails=` engine and `check_output()` can return
  `transformed_text` (redact-action rails); the validation pipeline ships
  the redacted text for plain-text outputs.
- The runtime negotiates the structured-output decoding mode per run and
  sends the strict-sanitized schema to capable providers (previously the
  raw schema was sent, which strict decoders such as OpenAI
  `strict: true` reject for open objects).
- **467 tests passing offline in ~2s; ruff clean**; seventeen runnable
  examples; 67 CI-gated VincioBench budgets.

## [0.6.0] - 2026-06-12

Agents & orchestration — the 0.6 roadmap milestone. Match the orchestration
frameworks on expressiveness, beat them on safety and observability:
multi-agent crews over a shared blackboard, durable stateful graphs with
checkpoint/resume/time-travel, first-class human-in-the-loop on graphs and
workflows, a declarative composition API with streaming node events, and
runtime backends that export to LangGraph and the OpenAI Agents SDK.

### Added

- **Multi-agent crews** — `Crew` / `app.crew(members=[...], process=...)`
  binds named `AgentRole`s (description, goal, keywords, `budget_fraction`)
  to bounded `AgentExecutor`s and runs them as a team: `sequential` (each
  member sees everything posted so far), `parallel` (bounded concurrent
  fan-out, dict of answers), and `hierarchical` (a manager decomposes the
  objective, delegates with a schema-validated plan, reviews the board, and
  either finishes or delegates follow-ups — with a deterministic
  keyword-routing fallback offline). Termination is guaranteed by
  construction: members run under a scaled share of the crew budget, the
  crew checks its budget before every delegation, and review rounds are
  capped at `max_rounds`. `CrewResult` carries per-member reports,
  `DelegationRecord`s, the blackboard snapshot, aggregated usage, and
  eval-ready `metrics()`.
- **Shared blackboard** — `Blackboard`: versioned, author-attributed shared
  working memory with per-key history, optional `blackboard.posted` events
  on the app event bus, prompt rendering (`as_context()`), and JSON
  `snapshot()` / `restore()` so crew coordination persists and replays.
- **Durable stateful graphs** — `StateGraph` / `app.graph()`: dict-state
  nodes (sync or async), static and conditional edges, optional per-key
  `reducers` for deterministic parallel-branch merges, and an optional
  Pydantic `state_schema` validated after every merge. `compile()` produces
  a `CompiledGraph` whose `Checkpointer` persists a checkpoint after every
  super-step on any `MetadataStore` (in-memory/SQLite/Postgres — `app.graph()`
  binds the app's store, so threads survive restarts): `resume(thread_id)`
  continues an interrupted thread, `history()` lists every checkpoint,
  `fork(checkpoint_id)` time-travels by branching a new thread that
  re-executes deterministically from that step, and `max_steps` bounds
  cyclic graphs. `astream()` yields node/checkpoint/interrupt/done events.
- **Human-in-the-loop** — pause graphs statically (`interrupt_before` /
  `interrupt_after` node lists) or dynamically from inside a node
  (`interrupt(state, payload)`); resume with a value and the paused node
  re-runs and receives it; `update_state(thread_id, values)` edits state as
  a new checkpoint before resuming. Workflow approval gates pause too:
  a gate with no `approval_fn` returns status `"paused"` with
  `pending_approvals`, and `workflow.resume(result, approvals={...})`
  continues without re-running done steps (edit the saved context to steer
  the continuation).
- **Declarative composition** — `compose(...)` / the `|` operator build
  typed pipelines from any mix of functions, agents, crews, workflows, and
  compiled graphs, normalizing results between steps (`AgentState` → final
  answer, `WorkflowResult`/`CrewResult` → output, `GraphResult` → state);
  `parallel(...)` fans out to named branches, `branch(router, routes)`
  routes by a function. `astream()` yields `NodeEvent`s
  (node_start/node_end/error/done) and every node emits a `compose_node`
  span.
- **Runtime backends** — `LangGraphBackend` exports a Vincio `StateGraph`
  to a LangGraph builder (nodes transfer as-is; edges, conditional edges,
  entry point, and `END` are translated) and `OpenAIAgentsBackend` exports
  agents and crews to OpenAI Agents SDK `Agent` objects (a crew becomes a
  manager agent with handoffs to every member; tools wrap via
  `function_tool`). Both import their runtime lazily and accept an injected
  module, so Vincio orchestrates without lock-in and the adapters test
  offline.
- **Observability** — new span types `crew`, `crew_agent`, `graph_node`,
  and `compose_node`; every crew member, graph node, and composed step is
  traced and scoreable like any other Vincio run.
- **VincioBench** — the `agent` family now also measures crew over-budget
  termination, full-crew success, delegation recording, interrupt→resume
  and fork-replay determinism (state must equal the uninterrupted run), and
  composition streaming coverage; six new `budgets.json` gates hold them in
  CI.
- **Docs & examples** — a new how-to guide
  (`docs/guides/orchestrate-agents.md`), expanded
  `docs/concepts/agents.md`, comparison write-ups for CrewAI and the OpenAI
  Agents SDK, a durable-graphs section in the LangChain/LangGraph
  comparison, and runnable examples `15_multi_agent_crew.py` and
  `16_durable_graph.py`.

### Changed

- **Workflow approval gates without an `approval_fn` now pause instead of
  failing** — `WorkflowResult.status` gains `"paused"` and
  `pending_approvals`; `arun(context=..., approvals=...)` /
  `aresume(previous, approvals=...)` continue a prior run, never re-running
  steps already done. Gates answered by a configured `approval_fn` behave
  exactly as before.
- `ContextApp.agent()` executor construction was factored into a shared
  builder reused by `app.crew()` (per-member tools/planner/model
  overrides); public behavior is unchanged.
- New error type `GraphError` (subclass of `AgentEngineError`) for graph
  definition and execution failures.
- **Pre-merge review hardening** — crew members built by `app.crew()`
  receive only their own (or the crew-level) tools, never the app-wide
  enabled set; per-member budget shares are clamped to what remains of the
  crew budget, and an explicit `budget_fraction=0.0` is honored; an
  approvals map can never bypass a configured `approval_fn`, and unknown
  approval names raise; a step failure beside a paused gate in the same
  level is terminal (compensation runs, the run is not reported paused);
  resumed workflow segments rebuild every non-`done` step result so
  compensated/failed steps never leak stale outputs; graph threads that
  ended at `max_steps` resume from their checkpoint (recompile with a
  higher bound), re-invoking a finished thread raises `GraphError` (fork it
  instead), and a dynamic interrupt mid-frontier re-queues the successors
  of siblings that already ran; `Crew` rejects unknown `process` values and
  `app.crew()` rejects unknown member fields; the LangGraph export gives
  routers exclusive edge precedence like the native engine; tracer
  trace/span cleanup tolerates abandoned streaming generators
  (`break` out of `astream`) without contextvar corruption.
- **426 tests passing offline in ~2s; ruff clean**; sixteen runnable
  examples; the VincioBench `agent` family holds the new orchestration
  guarantees under six additional CI-gated budgets.

## [0.5.0] - 2026-06-12

Evaluation, testing & observability — the 0.5 roadmap milestone. Make
evaluation and observability so good you stop reaching for an external
platform: metric parity with the eval specialists, unit-test ergonomics,
red-teaming, synthetic data, experiments with significance, a prompt
registry, sessions and feedback on traces, and a local viewer — all
provider-neutral, offline, and in-process.

### Added

- **Metric library expansion** — `faithfulness` (Ragas-style claim
  attribution), `answer_relevance` (penalizes evasive answers),
  `hallucination` (unsupported verifiable claims with **strict number
  checking** — "90 days" against evidence saying "30 days" fails; citation
  markers are stripped first), `toxicity` and `bias` (deterministic
  pattern-based rates), `summarization_quality`
  (min(coverage, faithfulness) against the source), and conversational
  metrics `knowledge_retention` (flags re-asking for facts the user already
  gave) and `conversation_relevance` (both read `context["messages"]`).
  All deterministic, offline, and usable as eval metrics, runtime
  evaluators, and test assertions.
- **G-Eval judge** — `GEvalJudge(provider, model=..., criteria=...)`
  auto-derives evaluation steps from plain-language criteria (cached for the
  judge's lifetime), scores on a 1–5 form-filling scale normalized to 0–1,
  approximates probability-weighted scoring with `samples > 1`, and
  `calibrate(pairs)` fits a linear correction against human labels
  (returns scale/offset/Pearson r) applied to future scores.
- **Testing ergonomics** — new `vincio.testing` package: `assert_eval`,
  `assert_grounded`, `assert_metric`, `assert_safe` raise AssertionErrors
  with the metric breakdown and offending output; quality metrics assert
  `>=`, rate metrics (`hallucination`, `toxicity`, ...) assert `<=`. A
  pytest plugin (registered via the `pytest11` entry point) adds the
  `vincio_snapshot` fixture and `--vincio-update-snapshots`; snapshots
  capture packet/trace *structure* with volatile fields (ids, timestamps,
  durations, hashes) normalized away, stored as JSON next to the tests.
- **Red-teaming & robustness** — `RedTeamSuite` sends 13 built-in probes
  (jailbreaks, prompt injections, PII/secret-leak probes, bias and toxicity
  provocations) at a `ContextApp` or any callable and judges responses
  deterministically: attack probes carry a canary token, leak probes run the
  secret scanner and PII detector, bias/toxicity probes reuse the new
  metrics. Reports separate `attack_success_rate` (output level) from
  `detector_coverage` (input-side injection detection); custom probes via
  `RedTeamProbe`. The injection detector gained `persona_without_rules` and
  `fake_authority` signals plus hardened override/exfiltration patterns —
  built-in probe coverage is 7/7 with no new false positives.
- **Synthetic data generation** — `SyntheticGenerator` bootstraps golden
  datasets from documents/chunks/text with difficulty mix (`easy` stated
  facts, `medium` cloze values, `hard` multi-hop across sources), coverage
  controls (round-robin over sources, near-duplicate dedupe), and full
  provenance (`metadata.source_ids`, source sentences in `rubric.facts` so
  grounding metrics work immediately). Deterministic offline templates by
  default; LLM-written questions when a provider is given, falling back to
  templates on failure.
- **Experiment tracking** — `ExperimentTracker` logs eval reports under
  experiment/variant (SQLite via the existing metadata store), `compare()`
  picks the best variant per metric (direction-aware: cost/latency/
  hallucination-style metrics minimize), `ablation()` reports deltas vs a
  baseline with p-values, and `ab_test(report_a, report_b, metric)` runs a
  paired t-test when reports share case ids, Welch's t-test otherwise —
  pure-Python t-distribution (regularized incomplete beta), no SciPy.
- **Prompt registry** — `PromptRegistry`: file-backed versioned prompt store
  keyed by `spec_hash` (re-pushing unchanged content is idempotent), tags
  that move between versions ("production", "candidate"), field-level and
  rendered diffs, `rollback()` that re-publishes an old version as a new
  head (history kept), and `link_eval()` attaching eval-run summaries to the
  exact version they measured. CLI: `vincio prompt push / versions / diff /
  rollback`.
- **Richer trace model** — traces carry `session_id` / `thread_id`
  (`app.run(..., session_id=...)` threads them through), `scores` (runtime
  evaluators attach metric scores to the eval span and the trace), and
  first-class `Feedback` (`trace.add_feedback`, `record_feedback(...,
  exporter=...)` persists updates; `vincio trace feedback`). Sessions are a
  derived view: `sessions_from_traces()` groups traces (deduping re-exported
  records) into `Session` objects with run/duration/error/score/feedback
  aggregates; `vincio trace sessions` lists them.
- **Traces become datasets** — `dataset_from_traces(traces,
  min_feedback_score=...)` curates captured runs into an eval dataset with
  full provenance (trace/run/session ids, scores); CLI:
  `vincio eval dataset golden.jsonl --min-feedback 0.5`.
- **OpenTelemetry GenAI semantic conventions** — the OTel exporter emits
  `chat {model}` / `execute_tool {tool}` span names with
  `gen_ai.operation.name`, `gen_ai.request.model`,
  `gen_ai.usage.input_tokens` / `output_tokens`,
  `gen_ai.response.finish_reasons`, `gen_ai.tool.name`, and
  `gen_ai.conversation.id` (sessions), alongside the full `vincio.*`
  attributes and span scores.
- **Local trace viewer** — `render_trace_text` / `render_session_text` (TUI
  tree with status glyphs, durations, scores, feedback; `vincio trace
  view`), `trace_to_html` / `session_to_html` (one self-contained static
  HTML file, inline CSS, no server or account; `vincio trace export
  [--session]`), and `trace_diff_html` (side-by-side visual diff;
  `vincio trace diff --html`).
- **Surface** — `vincio.evals` exports `GEvalJudge`, `SyntheticGenerator`,
  `RedTeamSuite` / `RedTeamProbe` / `BUILTIN_PROBES`, `ExperimentTracker` /
  `ab_test`, `dataset_from_traces`; `vincio.observability` exports
  `Session`, `Feedback`, `sessions_from_traces`, `record_feedback`, and the
  viewer functions; `vincio.prompts` exports `PromptRegistry` /
  `PromptVersion`; new `vincio.testing` package.
- **VincioBench `evals` family** — measures metric agreement on labeled
  examples, red-team judging on guarded vs naive targets, synthetic-data
  determinism and coverage, the significance machinery (detects a real
  shift, ignores a null one), session grouping, HTML self-containment,
  trace→dataset conversion, and G-Eval calibration — 13 new `budgets.json`
  gates hold the results in CI.
- Documentation: new observability concept guide and pytest testing guide,
  expanded evals concept guide, comparison write-ups for DeepEval and
  LangSmith/Langfuse, updated Ragas comparison; example
  `14_evaluation_observability.py`.

### Changed

- **OTel span names for model/tool spans changed** to the GenAI semantic
  conventions: `model_call:<name>` → `chat {model}`, `tool_call:<name>` →
  `execute_tool {tool}`. Dashboards or alerts keyed on the old span-name
  prefixes need updating; all `vincio.*` attributes (including
  `vincio.span_id`) are unchanged, and non-model/tool spans keep the
  `{type}:{name}` format.
- Model spans now record `input_tokens` (alongside `output_tokens`), and
  completed runs store their output (truncated) and eval scores on the
  trace, so traces are curatable into datasets.
- `JSONLExporter.load_all()` now returns the latest record per trace id
  (re-exports act as updates, e.g. after `record_feedback`).
- `EvalReport.diff()` is direction-aware: a rising `hallucination` /
  `toxicity` / `bias` / `unsupported_claim_rate` now counts as a regressed
  case (previously only falling scores did). Metric direction has a single
  source of truth: `vincio.evals.metrics.LOWER_IS_BETTER`.
- `EvidenceItem`-based grounding metrics accept reference context from
  `case.context["reference"]` / `["source"]` when a run carries no
  evidence.
- **367 tests passing offline in ~2s; ruff clean**; fourteen runnable
  examples; 48 VincioBench budget gates.

## [0.4.0] - 2026-06-12

Memory & personalization — the 0.4 roadmap milestone. Personalization
without the failure mode of stale, ungrounded memories: every memory
carries confidence, provenance, decay, and conflict resolution, and is
utility-scored against the task before it ever enters a packet.

### Added

- **Personalization APIs** — `remember()` / `recall()` ergonomics over the
  L0–L5 layers, on both `MemoryEngine` and `ContextApp` (`app.remember(...,
  user_id="u1")` auto-creates the engine). Scope and memory type are
  inferred (session > agent > user > tenant; preference/goal/decision/fact
  classification). New `MemoryScope.AGENT` gives every agent durable memory
  of its own, and `ScopedMemory` handles (`memory.for_user("u1")`,
  `for_agent`, `for_session`, `for_tenant`) bind one owner for
  `remember` / `recall` / `forget` / `items` / `export`.
- **Hybrid memory recall** — `MemoryEngine.asearch()` fuses lexical and
  vector relevance (`(1−w)·lexical + w·cosine` over any `Embedder`, offline
  hash embedder by default, content-addressed vector cache) with graph
  adjacency (memories linked to the task's entities get a boost) in one
  scored, scope- and privacy-filtered query; `search()` stays as the sync
  wrapper. The runtime's memory step extracts task entities and recalls
  hybrid by default (`memory.hybrid_recall`, `memory.vector_weight`).
- **Consolidation tiers** — `MemoryConsolidator` (and
  `await memory.consolidate(session_id, user_id=...)`): episodic session
  memories summarize into semantic memories promoted to user/agent scope,
  deduplicate (the survivor absorbs confirmations and records
  `merged_from`), and retain full provenance — promoted items carry
  `consolidated_from`, episodes are archived with `consolidated_into`,
  never silently dropped. `promote_aged_episodes()` runs the background
  tier transition.
- **Forgetting & hygiene** — per-scope TTL defaults applied on write
  (`memory.ttl_days`, sessions default to 30 days) with expired items
  excluded from recall; importance-weighted retention in `decay_pass()`
  (heavily used, confirmed, stable preferences/decisions survive longer —
  `importance_score`, `memory.retention_weight`); and user-driven
  `edit` / `forget` / `export_owner_data` / `erase_owner_data`
  (GDPR-style access, rectification, portability, erasure) flowing through
  the hash-chained audit log as `memory_edit` / `memory_delete` /
  `memory_export` / `memory_erase` entries.
- **Memory eval harness** — `vincio.memory.evaluate_memory` measures recall
  precision, recall@k, contradiction rate, staleness, and personalization
  lift (owner-scoped vs anonymous recall) against labeled
  `MemoryEvalCase`s; the VincioBench `memory` family runs it plus
  consolidation/TTL checks, gated in CI by eleven new `budgets.json`
  entries.
- **Run write-back** — step 16 is now governed by `memory.write_back`
  (`input` | `evidence` | `tools`): cited evidence and successful tool
  results write back as *candidate* memories with provenance
  (`origin` / `source_id` / `tool_name`), carrying a status penalty in
  recall until confirmed (restatement or `confirm()` promotes them to
  active).
- **Surface** — CLI `vincio memory remember | recall | forget | export |
  consolidate | decay`; server endpoints `POST /v1/memory/consolidate`,
  `GET /v1/memory/export`, `GET /v1/memory/stats`,
  `DELETE /v1/memory/{id}`; `extract_entities` is now public in
  `vincio.retrieval.chunking`; new docs (rewritten memory concepts page, a
  Mem0 comparison) and `examples/13_memory_personalization.py` (offline).

### Changed

- `MemoryEngine` accepts `embedder`, `vector_weight`, `retention_weight`,
  `ttl_days`, and `audit`; `app.add_memory()` wires the app's embedder and
  audit log automatically. Search components now report `lexical`,
  `vector`, `graph`, and `status` alongside the existing factors.
- `MemoryEngine.search()` includes `candidate`-status memories with a 0.7
  status weight, and restatements re-activate confirmed candidates.
- 301 tests passing offline (~2s); ruff clean.

## [0.3.0] - 2026-06-12

Retrieval & RAG superiority — the 0.3 roadmap milestone. Every advanced
retrieval technique behind one `Index` interface, fused in one weighted RRF,
budgeted and cited inside the compiled packet, and measured by CI-gated
benchmarks.

### Added

- **Learned sparse retrieval** — `SparseIndex`, an inverted impact index
  scored by SPLADE-style dot products, behind the same `Index` protocol as
  BM25/dense so it fuses in the existing weighted-RRF merge. Encoders:
  `LocalImpactEncoder` (offline, deterministic: sublinear tf + morphological
  stem expansion) and `CallableSparseEncoder` (adapter for served SPLADE /
  uniCOIL / ELSER models).
- **Late-interaction retrieval** — `LateInteractionIndex` with ColBERT-style
  per-token MaxSim scoring over any `Embedder` (offline hash embedder by
  default, ColBERT checkpoints behind the same protocol). `compressed=True`
  enables PLAID-style two-stage search: deterministic k-means centroid
  codes, candidate generation over inverted centroid lists, exact rerank of
  survivors. Token-vocabulary vector caching keeps indexing cheap.
- **Advanced indexing** — new chunking strategies: `sentence_window` (score
  the sentence, cite the ±2-sentence window — the engine swaps the window in
  at evidence time), `hierarchical`/`parent_document` (small children linked
  to large parents), and `contextual` (situating prefix per chunk).
  `AutoMergingIndex` wraps any index and merges sibling child hits back into
  their parent; `contextualize_chunks()` writes LLM chunk prefixes
  (contextual retrieval) with a heuristic offline fallback.
- **Query understanding** — `QueryUnderstanding` strategies: HyDE
  (hypothetical answer passage as a search probe), multi-query expansion,
  decomposition for multi-hop, and step-back prompting. LLM-backed with
  deterministic offline fallbacks; expansions are recorded on the
  `QueryPlan`, fused with per-strategy RRF weights, and surfaced in
  retrieval metadata/traces. Configure per engine
  (`RetrievalEngine(query_strategies=[...])`), per call
  (`retrieve(strategies=[...])`), or app-wide
  (`retrieval.query_strategies`).
- **GraphRAG** — `detect_communities` (deterministic label propagation over
  the entity graph), `Community` hierarchy (communities of communities),
  extractive community summaries with an LLM hook, and `GraphRAG` retrieval
  with global vs local routing: entity questions walk graph paths,
  corpus-level questions retrieve community summaries that carry provenance
  to their member chunks.
- **Incremental & live indexes** — `LiveIndex` wraps any index with upsert
  semantics, per-entry TTLs, lazy `purge_expired()`, and `indexed_at`
  freshness stamps; the retrieval engine surfaces `indexed_at` and
  `age_days` in evidence metadata. `VectorIndex.migrate(new_embedder)`
  re-embeds in place — an embedding-model migration without re-chunking or
  rebuilding.
- **Connector hub** — new `vincio.connectors` package: `web`, `github`,
  `sql` (SQLite built in, any DB-API connection), `s3` (`vincio[s3]`),
  `gcs` (`vincio[gcs]`), `notion`, `confluence`, and `slack` connectors,
  all returning provenance-tracked `Document`s; a `connect()` factory and
  `register_connector()` plugin point; `app.add_source(connector=...)`
  loads, chunks, and indexes in one call. REST connectors accept injected
  httpx clients (offline-testable); cloud connectors accept injected
  boto3/GCS clients.
- **App retrieval modes** — `add_source(retrieval=...)` now also accepts
  `sparse`, `late_interaction`, and `hybrid_full` (BM25 + dense + sparse +
  late interaction in one fusion).
- **VincioBench** — the `rag` family now compares every retrieval mode
  (bm25, dense, sparse, late_interaction, late_interaction_plaid, hybrid,
  hybrid_full, hybrid_full + query understanding) on recall@3/MRR and
  exercises GraphRAG community building; new `budgets.json` gates hold each
  mode at recall@3 ≥ 0.8 and verify GraphRAG produces communities and
  global evidence.
- **Docs & examples** — rewritten retrieval concepts page, a new
  connectors guide (`docs/guides/connectors.md`), a new RAGatouille/ColBERT
  comparison (`docs/comparisons/ragatouille.md`), an updated LlamaIndex
  comparison, and `examples/12_advanced_rag.py` (sparse + late-interaction
  fusion, query understanding, auto-merging, GraphRAG routing, live-index
  TTL, SQL connector → full app — offline).

### Changed

- `QueryPlan` gains `expansions`; `RetrievalResult.metadata` reports the
  strategies used; evidence from sentence-window chunks carries the window
  text plus a `matched_sentence` marker.
- 277 tests passing offline (~2s); ruff clean.

## [0.2.0] - 2026-06-12

Performance & core hardening — the 0.2 roadmap milestone. The spine is now
fast, streaming, measured, and regression-gated.

### Added

- **End-to-end streaming** — `ContextApp.astream` (and sync `stream`) runs
  the full 17-step pipeline with real provider token streaming:
  `RunStreamEvent`s for pipeline stages, text deltas, incremental
  partial-JSON output (structure-only, never invents content), tool
  activity, and a terminal `done` with the validated `RunResult`. The model
  span records `ttft_ms`; the server `/stream` endpoint now emits real
  deltas over SSE instead of chunking the finished answer. `MockProvider`
  streams in genuine chunks so the path is exercised offline.
- **Async-first hot paths** — memory recall, file ingestion, and retrieval
  run concurrently per run; retrieval fans out every (query × index) pair;
  tool calls within a model round execute concurrently (bounded by
  `performance.tool_parallelism`). New `vincio.core.concurrency` module
  (`gather_bounded`, `map_bounded`): order-preserving, semaphore-bounded,
  first-failure-cancels-the-group fan-out.
- **Cancellation & deadlines** — cancelling `arun`/`astream` cancels every
  in-flight subtask; `Budget.max_latency_ms` is enforced as a hard deadline
  (the run fails with a budget error instead of hanging); cancelled runs
  persist with status `cancelled`.
- **Incremental & cached compilation** — content-addressed caches, on by
  default, keyed over every input that affects the output:
  `PromptCompileCache`, `ChunkCache` (keyed by document *content*, with
  provenance restored per requesting document), and `ContextCompileCache`.
  `ContextCompiler.recompile(previous, add_evidence=, remove_evidence_ids=, ...)`
  re-runs selection over retained inputs for cheap packet edits; the
  lexical scorers (`_terms`/`_shingles`) are memoized, removing the
  re-tokenization cost from the O(n²) dedupe/conflict passes. All caches
  invalidate through the existing tag-based invalidation manager.
- **Zero-copy Context Packet** — `slim_packets` mode references evidence
  text by content hash (text lives once, on the IR) with lazy
  materialization (`packet.evidence_text(id)`, `packet.materialize()`);
  `packet.iter_json()` streams serialization chunk by chunk;
  `packet.approx_size_bytes()` reports size without building the blob.
- **Throughput primitives** — connection-pooled provider transport
  (`httpx.Limits`, configurable pool sizes) with provider instances reused
  across runs; `CoalescingProvider` dedupes identical in-flight `generate`
  calls (on by default via `performance.coalesce_requests`);
  `ProviderEmbedder` splits large inputs into bounded concurrent batches;
  `BatchingEmbedder` micro-batches concurrent embed calls into one
  provider round-trip; `CachedEmbedder` is now thread-safe,
  content-addressed (SHA-256 keys), and accepts a persistent backend.
- **Benchmark gates in CI** — new VincioBench `perf` family (compile/
  retrieval/run latency percentiles, cache speedups, concurrent
  throughput, streaming TTFT); `benchmarks/budgets.json` +
  `benchmarks/check_budgets.py` fail the build on regression; new CI
  `bench` job uploads the report. `benchmarks/profile_stages.py` gives a
  per-stage breakdown from trace spans plus cProfile output for
  flamegraphs.
- **Config** — new `performance` section (`max_concurrency`,
  `tool_parallelism`, `embed_batch_size`, `embed_window_ms`,
  `coalesce_requests`, `max_connections`, `max_keepalive_connections`,
  `slim_packets`, `partial_parse_min_chars`) and new `cache` flags
  (`prompt_compile_cache`, `chunk_cache`, `context_compile_cache`).
- Docs: new [performance & streaming guide](docs/guides/performance.md);
  API/config references updated. New example
  `11_streaming_performance.py`. 34 new tests (229 total, offline).

### Fixed

- `PromptCompiler.compile` no longer temporarily mutates shared options to
  toggle schema rendering — it was a data race under concurrent compiles.

## [0.1.0] - 2026-06-12

Initial public release.

### Added

- **Prompt engine** — typed `PromptSpec`, AST, cache-aware compiler, linter, variant generation.
- **Context compiler** — candidate scoring, token budgeting, compression/distillation, evidence
  ledger, and excluded-candidate reports.
- **Engines** — input (normalization, classification, routing), documents (loaders, parsers, OCR,
  multimodal), retrieval (hybrid BM25 + vector RRF, rerankers, graph, reasoning), memory (layered
  store, decay, conflict resolution, graph), tools (permissioned runtime, sandbox), agents
  (bounded DAG, ReAct, handoffs), workflows (deterministic DAG), and output (schemas, robust
  parsers, validation, principled repair).
- **Evaluation** — datasets, metrics, judges, runner, regression gates, and reports.
- **Optimization** — gated prompt / context / routing / cache search.
- **Observability** — traces, spans, JSONL/OTel exporters, cost tracking.
- **Security** — PII and secret handling, prompt-injection defense, RBAC/ABAC access control,
  deterministic policy engine, and audit logging.
- **Caching** — response / retrieval / packet / semantic caches with invalidation.
- **Storage adapters** — SQLite, Postgres (pgvector), Qdrant, Neo4j, Redis, DuckDB.
- **Providers** — OpenAI, Anthropic, Google, Mistral, local, and a deterministic offline mock.
- **Surfaces** — FastAPI server (API key + JWT auth) and an argparse CLI.
- 195 offline tests, 10 runnable examples, documentation, and the VincioBench benchmark suite.

[1.0.0]: https://github.com/Ohswedd/vincio/releases/tag/v1.0.0
[0.2.0]: https://github.com/Ohswedd/vincio/releases/tag/v0.2.0
[0.1.0]: https://github.com/Ohswedd/vincio/releases/tag/v0.1.0
