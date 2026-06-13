# Reference: CLI

```text
vincio init [path] [--project NAME] [--force]
    Scaffold vincio.yaml, app.py, and golden/basic.jsonl.

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
```
