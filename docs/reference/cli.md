# Reference: CLI

```text
vincio init [path] [--template minimal|rag|agent|eval] [--provider NAME]
        [--project NAME] [--force]
    Scaffold a project from a template: vincio.yaml (with a JSON Schema editor
    hint), app.py, vincio.schema.json, and a golden set. rag adds docs/ + a
    grounded app; agent adds a tool; eval adds a dataset + run instructions.

vincio config schema [--output FILE]
    Emit the vincio.yaml JSON Schema (from the typed VincioConfig) for editor
    completion and validation.

vincio config validate [PATH]
    Validate a vincio config file (or the nearest one); exits non-zero on error.

vincio config show [PATH]
    Print the effective merged configuration as YAML.

vincio config migrate [PATH] [--output FILE] [--dry-run] [--check]
    Upgrade a vincio.yaml to the current schema version, reporting each applied
    migration step. --check exits non-zero if a migration is pending (CI gate)
    without writing; --dry-run previews; --output writes elsewhere. A leading
    # yaml-language-server schema hint is preserved. (Stale files also migrate
    in memory automatically on load.)

vincio doctor [PATH] [--json]
    Scan a project for deprecated-API usage (driven by the same stability_of
    metadata the library marks its surface with, each finding names the
    replacement and removal version) and for a vincio.yaml behind the current
    schema. Exits non-zero if any actionable issue is found.

vincio migrate TARGET [PATH] [--write] [--check] [--json]
    Rewrite a project's source for a major-version upgrade (the code-surface
    analogue of `config migrate`). A static, ast-based codemod, it never
    imports or runs your code, driven by a declarative per-major rename table,
    rewriting only the exact identifier tokens a rename touches. Default is a dry
    run that prints the plan; --write applies the rewrites in place; --check
    exits non-zero if a migration is available (CI gate) without writing; --json
    emits the plan as JSON. TARGET is the major to migrate to (e.g. 4.0). The 4.0
    table is empty, a clean 3.x → 4.0 upgrade needs no source changes; see
    MIGRATION.md.

vincio packs list
    List the available domain packs (support, engineering, finance, legal).

vincio packs show NAME
    Show a pack's role, objective, policies, evaluators, and output schema.

vincio plugins list
    List installed third-party plugins (providers, embedders, stores,
    connectors, chunkers, rerankers, judges, metrics, packs) discovered via
    entry points, with each one's distribution, version, and status
    (available / loaded / incompatible / error) under the versioned plugin API.

vincio tui [--traces-dir DIR] [--db FILE]
    Interactive terminal inspector for runs, traces, and memory.

vincio run APP.py --input "..." [--file F]... [--tenant T] [--user U]
    Run the app once and print status, trace id, cost, and output.

vincio batch APP.py --input X --input Y [--input-file lines.txt]
        [--discount 0.5] [--output results.json]
    Run a set of inputs through the provider's Batch API at ~50% cost
    (--discount). Inputs come from repeated --input flags and/or one per line
    from --input-file. Prints N/M succeeded, total cost, and a per-result
    trace id; --output writes the results as JSON. Exits non-zero if any
    request failed.

vincio cost report --by tenant|feature|user|model|provider|run
        [--db .vincio/vincio.db] [--json]
    Roll up attributed model cost by a dimension from the metadata store's
    cost_events (--db). Prints per-key cost, calls, and tokens with a total;
    --json emits the report as JSON.

vincio eval run DATASET.jsonl --app APP.py
        [--metric NAME]... [--gate "metric=>= 0.9"]...
        [--compare baseline.json] [--output report.json] [--concurrency N]
    Run an eval; exits non-zero when gates fail.

vincio eval report REPORT.json|DIR
    Print a saved report (latest in a directory).

vincio eval dataset OUTPUT.jsonl [--traces-dir DIR] [--name N] [--min-feedback X]
        [--group-by-session]
    Curate captured traces into an eval dataset (provenance + scores ride along).
    --group-by-session stitches a session's traces into one multi-turn case.

vincio eval drift BASELINE.json CURRENT.json [--metric NAME]... [--threshold X]
        [--output drift.json]
    Report per-metric drift between two eval reports; exits non-zero on drift.

vincio eval annotate LABELS.jsonl [--threshold X] [--bins N]
    Report human↔judge Cohen's κ from {judge, human} score pairs; exits
    non-zero until κ clears the threshold (the judge's CI-gating bar).

vincio eval regress DATASET.jsonl --app APP.py --candidate-model Y
        [--baseline-model X] [--metric NAME]... [--quality-metric M]
        [--alpha 0.05] [--repeats N] [--no-flake-quarantine] [--output report.json]
    Swap only the model and report a statistically grounded regression:
    per-metric significance, the cost/latency trade, and worst-regressed slices.
    Exits non-zero on a significant quality regression.

vincio eval suite list [--json]
    List the open-evaluation-plane catalog grouped by niche, with each benchmark's
    primary metric and the tiers it supports (S always; R/L when it has a loader).

vincio eval suite run [BENCHMARK|NICHE|all]... [--app APP.py] [--tier static|recorded|live]
        [--sample N] [--concurrency N] [--format text|markdown|html|json|csv|pdf]
        [--output PATH] [--store DSN] [--version TAG]
    Run public benchmarks from the open evaluation plane (e.g. knowledge.mmlu, or a
    whole niche, or all). The default tier `static` replays the bundled fabricated
    fixtures fully offline; `recorded`/`live` need a dataset (and, for live, --app).
    Every number carries its provenance tier; a lower tier cannot print a higher
    tier's label. For a Live run against a state-of-the-art model over a real
    dataset, see `benchmarks/eval_live.py`.

vincio eval suite leaderboard --store DSN [--model NAME]... [--limit N] [--json]
    Rank persisted suite runs (one row per model) over a shared benchmark set.

vincio eval suite report RUN.json|RUN_ID [--store DSN]
        [--format markdown|html|json|csv|pdf] [--output PATH]
    Render a saved suite run to any format, citing the exact scored items.

vincio eval suite compare RUN_A RUN_B --store DSN [--json]
    Diff two persisted suite runs: overall delta plus per-benchmark regressions
    and improvements.

vincio bench list [--json]
    List all three tracks' catalogs at a glance: the model track's benchmarks and
    niches, the six uplift benchmarks, and the feature contests.

vincio bench model [BENCHMARK|NICHE|all]... [--app APP.py]
        [--tier static|recorded|live] [--sample N] [--concurrency N]
        [--format text|markdown|html|json|csv|pdf] [--output PATH] [--store DSN]
        [--version TAG]
    Track 1 — a model on the public benchmarks (an alias for `eval suite run`).

vincio bench uplift [BENCHMARK|all]... [--tier static] [--format text|markdown] [--json]
    Track 2 — the same model direct vs routed through Vincio, scored twice by the
    identical scorer; prints the per-benchmark delta. This subcommand runs the
    offline mockup (tier `static`); a live uplift run is driven from Python with
    real `direct`/`vincio` targets.

vincio bench feature [CONTEST|CAPABILITY|all]... [--format text|markdown] [--json]
    Track 3 — a Vincio feature vs the same feature in a competitor library, measured
    live on this machine. A missing competitor is reported skipped, never fabricated;
    the contest then drops to the Static tier.

vincio prompt lint PATH
    Lint prompt spec YAML files (PROMPT001–PROMPT009); exits non-zero on errors.

vincio prompt compile SPEC.yaml [--format markdown|xml|json|minimal] [--task "..."]
    Compile and print a prompt with hashes, token count, and cacheability.

vincio prompt push SPEC.yaml [--name N] [--tag T]... [--message M] [--registry DIR]
vincio prompt versions NAME [--registry DIR]
vincio prompt diff NAME V_A V_B [--rendered] [--registry DIR]
vincio prompt rollback NAME [--to V] [--registry DIR]
    Versioned prompt registry: push (idempotent on content), list versions
    with tags/messages/eval links, diff two versions, re-publish an old
    version as the new head.

vincio trace show TRACE_ID  [--traces-dir DIR]
vincio trace view TRACE_ID  [--traces-dir DIR]
vincio trace replay TRACE_ID [--traces-dir DIR]
vincio trace diff TRACE_A TRACE_B [--traces-dir DIR] [--html OUT.html]
    Inspect (view: TUI tree with scores + feedback), extract a replay plan
    from, or diff traces (--html writes a visual side-by-side diff).

vincio trace export TRACE_ID [--session] [--output OUT.html] [--traces-dir DIR]
    Write a self-contained static HTML page for a trace (or, with --session,
    a whole session), no server, no account.

vincio trace sessions [--traces-dir DIR]
    List sessions with run counts, durations, error rates, scores, feedback.

vincio trace feedback TRACE_ID [--key K] [--score X] [--comment "..."] [--user U]
    Attach feedback to a stored trace (persisted as an update).

vincio trace verify-recording PATH
    Verify a causal recording's replay fidelity offline (no app needed): check
    every recorded edge against its content address, confirm the fidelity
    digest, and print the inspection summary. Exits non-zero on failure. A
    recording is produced by `Recorder(app).record(...).save(path)` and replayed
    byte-for-byte with `Replayer(app).replay(recording)`.

vincio optimize run --app APP.py --dataset DATASET.jsonl
        [--target quality|groundedness|cost|latency]
        [--budget N] [--subset N] [--output winning.yaml]
    Prompt-variant optimization with gated promotion.

vincio optimize reflective --app APP.py --dataset DATASET.jsonl
        [--strategy reflective|mipro] [--target quality|groundedness|cost]
        [--budget N] [--minibatch N] [--seed N] [--apply] [--output winning.yaml]
    GEPA-style reflective optimization: reads eval failures, reflects on
    why the prompt lost, proposes targeted edits, and evolves a Pareto frontier
    under a hard rollout budget. --strategy mipro switches to joint
    instruction+example proposal; --apply installs the winner on the app.

vincio loop run --app APP.py [--dataset DATASET.jsonl | --min-feedback X]
        [--gate "metric=>= 0.9"]... [--budget N] [--subset N]
        [--tag production] [--experiment NAME] [--dry-run] [--reflective]
    One closed-loop cycle: trace → dataset → eval → optimize → promote.
    Without --dataset, curates the dataset from captured traces (feedback-
    filtered). The promoted version is pushed to the prompt registry,
    tagged, eval-linked, applied to the app, and audited; --dry-run
    reports the decision without acting on it; --reflective uses the
    GEPA-style reflective optimizer.

vincio distill --output TRAIN.jsonl [--traces-dir DIR]
        [--format openai|anthropic] [--min-feedback X] [--min-support X]
        [--max-examples N] [--allow-ungrounded]
    Curate captured traces into grounded fine-tuning JSONL: feedback-
    filtered, grounding-checked against cited evidence, deduped, with full
    provenance. Ungrounded examples are dropped unless --allow-ungrounded.

vincio index build PATH [--db FILE] [--chunking STRATEGY] [--chunk-size N]
    Load, chunk, and persist documents into a SQLite index store.

vincio memory inspect [--user U] [--db FILE] [--limit N]
    List stored memories.

vincio memory remember CONTENT [--user U] [--agent A] [--session S] [--tenant T]
        [--scope SCOPE] [--type TYPE] [--db FILE]
    Write one memory; scope and type are inferred when omitted.

vincio memory recall QUERY [--user U] [--agent A] [--session S] [--tenant T]
        [--top-k N] [--db FILE]
    Scored hybrid (lexical + vector + graph) recall.

vincio memory forget MEMORY_ID [--reason R] [--db FILE]
    Delete one memory; the reason lands in the audit log.

vincio memory export --owner OWNER [--output FILE] [--db FILE]
    GDPR-style export of every memory an owner has (audited).

vincio memory consolidate SESSION_ID [--user U] [--db FILE]
    Episodic→semantic consolidation for a session, with provenance.

vincio memory decay [--db FILE]
    Run a decay/TTL pass (importance-weighted retention).

vincio audit verify [PATH] [--json]
    Verify the SHA-256 hash chain of a persisted audit JSONL log offline
    (default .vincio/audit/audit.jsonl). Detects post-restart tampering and
    pinpoints the first broken line; exits non-zero if the chain is broken.

vincio governance card APP [--kind model|system] [--format vincio|open_model_card|ai_card] [--output FILE]
    Generate a model or system card (machine-readable) from the live app config.

vincio governance report APP [--red-team] [--full] [--markdown] [--output FILE]
    Emit the OWASP LLM 2025 / OWASP Agentic / NIST AI RMF / MITRE ATLAS coverage
    matrix. --red-team runs the red-team suite for behavioural evidence; --full
    emits every control; --markdown emits an auditor-ready table.

vincio governance aibom APP [--output FILE]
    Generate an AI bill of materials (CycloneDX 1.6) for the model, embedder,
    reranker, and any pinned datasets/prompts, with SHA-256 hash slots.

vincio governance lineage APP SOURCE [--output FILE]
    Trace a source's lineage chain (documents → chunks → evidence → runs).

vincio governance erase APP SOURCE
    Right-to-erasure: purge a source from every index, memory, and cache,
    logged on the hash-chained audit log. Idempotent.

vincio mcp tools (--command "CMD" | --url URL) [--resources] [--json]
    Connect to an MCP server (stdio via --command, or Streamable HTTP via
    --url) and list its tools (and, with --resources, its resources).

vincio mcp add APP --name NAME (--command "CMD" | --url URL) [--resources]
    Connect an MCP server to the ContextApp in APP and register its tools
    (namespaced NAME.<tool>); prints the registered tools.

vincio mcp serve APP [--name NAME]
    Expose the ContextApp in APP as an MCP server over stdio (reads JSON-RPC
    on stdin). Tools/resources/prompts are served through the permissioned,
    audited runtime.

vincio serve [--app FILE ...] [--config vincio.yaml] [--host H] [--port P]
    Launch the HTTP API (FastAPI + uvicorn) serving one or more ContextApps,
    with /v1/health, /v1/health/ready, /v1/metrics (Prometheus), real-token SSE
    streaming, and graceful shutdown. Configure server.redis_url for coherent
    rate-limit/idempotency state across workers.

vincio providers list [--provider NAME] [--json]
    List the model registry catalog (tier, lifecycle, pricing, successor).

vincio providers lifecycle [--app APP.py] [--model ID]... [--as-of YYYY-MM-DD]
        [--warn-within-days N] [--json]
    Scan pinned models (the app's, or --model ids) for sunset and propose
    migrations off deprecated/retired ones; exits non-zero on a warn/critical
    alert.

vincio providers discover PROVIDER [--mark-missing-deprecated] [--json]
    Reconcile a provider's live model list into the registry (offline-safe,
    the shipped catalog stands when no endpoint is reachable).

vincio providers regress --app APP.py --candidate-model Y [--baseline-model X]
        [--dataset DATASET.jsonl] [--trace TRACE_ID]... [--traces-dir DIR]
        [--gate "metric=>= 0.9"]... [--quality-metric M] [--alpha 0.05] [--repeats N]
    Gate a model swap: replay golden traces + an eval/cost/latency/behavioral
    diff with significance; exits non-zero on a FAIL verdict.

vincio registry coverage [--as-of YYYY-MM-DD] [--json]
    Prove the shipped model catalog is complete, honest, fresh, and
    routing-stable: every provider default + capability-heuristic family +
    openai_compat preset resolves to a non-sparse, priced profile, no GA billable
    model silently bills $0, no price has drifted past the freshness horizon, and
    the canonical router/cascade/energy picks are unchanged. Freshness is
    evaluated against the catalog's release date by default (--as-of overrides it
    for what-if checks; it never reads the wall clock). Exits non-zero on a gap.

vincio registry sync [PROVIDER] [--out OVERLAY.json] [--json]
    Review-only: diff a provider's live list_models() against the shipped catalog
    and emit a candidate overlay of models that need a human-set price and
    capabilities (default: openai/anthropic/google/mistral). Never mutates the
    catalog. Needs the providers' clients (extra `vincio[registry-sync]`) and
    credentials; offline-safe (a provider it cannot reach is skipped).

vincio docs map [--check] [--json]
    Regenerate the connected-docs artifacts from the doc graph (vincio._docmap):
    the capability map (docs/reference/capability-map.md), the learning path
    (docs/learning-path.md), the api.md app-method index, the single-sourced
    Related cross-link block on every concept and guide, and llms.txt. With
    --check it reports stale artifacts and exits non-zero without writing (CI
    gate). Dependency-free.

vincio docs check [--json] [--limit N]
    Run the docs-graph check: link integrity (every internal link resolves, path
    and anchor), capability-map coverage (every public app.* verb is mapped and
    documented in api.md, every concept reaches a guide + example + reference),
    navigation reachability (every concept and guide carries a current Related
    block and the generated pages are current), no orphans, and llms.txt
    freshness. Exits non-zero on any failure. Dependency-free.

vincio docs serve [--host H] [--port P]
    Serve the docs locally with a generated index and the docs-graph report.
    Renders Markdown to HTML with the richer renderer from `vincio[docs]`
    (markdown-it-py) when installed, and falls back to serving raw Markdown over
    the standard library otherwise.
```
