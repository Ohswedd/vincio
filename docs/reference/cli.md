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

vincio prompt lint PATH
    Lint prompt spec YAML files (PROMPT001–PROMPT009); exits non-zero on errors.

vincio prompt compile SPEC.yaml [--format markdown|xml|json|minimal] [--task "..."]
    Compile and print a prompt with hashes, token count, and cacheability.

vincio trace show TRACE_ID  [--traces-dir DIR]
vincio trace replay TRACE_ID [--traces-dir DIR]
vincio trace diff TRACE_A TRACE_B [--traces-dir DIR]
    Inspect, extract a replay plan from, or structurally diff traces.

vincio optimize run --app APP.py --dataset DATASET.jsonl
        [--target quality|groundedness|cost|latency]
        [--budget N] [--subset N] [--output winning.yaml]
    Prompt-variant optimization with gated promotion.

vincio index build PATH [--db FILE] [--chunking STRATEGY] [--chunk-size N]
    Load, chunk, and persist documents into a SQLite index store.

vincio memory inspect [--user U] [--db FILE] [--limit N]
    List stored memories.
```
