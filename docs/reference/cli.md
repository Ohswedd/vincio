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

vincio packs list
    List the available domain packs (support, engineering, finance, legal).

vincio packs show NAME
    Show a pack's role, objective, policies, evaluators, and output schema.

vincio tui [--traces-dir DIR] [--db FILE]
    Interactive terminal inspector for runs, traces, and memory.

vincio run APP.py --input "..." [--file F]... [--tenant T] [--user U]
    Run the app once and print status, trace id, cost, and output.

vincio eval run DATASET.jsonl --app APP.py
        [--metric NAME]... [--gate "metric=>= 0.9"]...
        [--compare baseline.json] [--output report.json] [--concurrency N]
    Run an eval; exits non-zero when gates fail.

vincio eval report REPORT.json|DIR
    Print a saved report (latest in a directory).

vincio eval dataset OUTPUT.jsonl [--traces-dir DIR] [--name N] [--min-feedback X]
    Curate captured traces into an eval dataset (provenance + scores ride along).

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
    a whole session) — no server, no account.

vincio trace sessions [--traces-dir DIR]
    List sessions with run counts, durations, error rates, scores, feedback.

vincio trace feedback TRACE_ID [--key K] [--score X] [--comment "..."] [--user U]
    Attach feedback to a stored trace (persisted as an update).

vincio optimize run --app APP.py --dataset DATASET.jsonl
        [--target quality|groundedness|cost|latency]
        [--budget N] [--subset N] [--output winning.yaml]
    Prompt-variant optimization with gated promotion.

vincio loop run --app APP.py [--dataset DATASET.jsonl | --min-feedback X]
        [--gate "metric=>= 0.9"]... [--budget N] [--subset N]
        [--tag production] [--experiment NAME] [--dry-run]
    One closed-loop cycle: trace → dataset → eval → optimize → promote.
    Without --dataset, curates the dataset from captured traces (feedback-
    filtered). The promoted version is pushed to the prompt registry,
    tagged, eval-linked, applied to the app, and audited; --dry-run
    reports the decision without acting on it.

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
```
