"""Vincio CLI.

Commands::

    vincio init
    vincio run app.py --input "..."
    vincio eval run golden.jsonl --app app.py
    vincio eval report <report.json>
    vincio prompt lint prompts/
    vincio prompt compile prompt.yaml
    vincio trace show <trace_id>
    vincio trace replay <trace_id>
    vincio trace diff <trace_a> <trace_b>
    vincio optimize run --app app.py --dataset golden.jsonl --target groundedness
    vincio index build ./docs
    vincio memory inspect --user u1
    vincio memory remember "Prefers concise answers" --user u1
    vincio memory recall "answer style" --user u1
    vincio memory forget <memory_id>
    vincio memory export --owner u1
    vincio memory consolidate <session_id> --user u1
    vincio memory decay
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from ..core.errors import VincioError
from ..core.utils import json_dumps

__all__ = ["main", "build_parser"]


def _fail(message: str, code: int = 1) -> int:
    print(f"error: {message}", file=sys.stderr)
    return code


def _load_app(path: str):
    """Import a python file and find the ContextApp instance."""
    from ..core.app import ContextApp

    module_path = Path(path)
    if not module_path.is_file():
        raise VincioError(f"app file not found: {path}")
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise VincioError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_path.stem] = module
    spec.loader.exec_module(module)
    candidate = getattr(module, "app", None)
    if isinstance(candidate, ContextApp):
        return candidate
    for value in vars(module).values():
        if isinstance(value, ContextApp):
            return value
    raise VincioError(f"no ContextApp instance found in {path} (expose one as `app`)")


def _load_trace(trace_id: str, traces_dir: str):
    from ..observability.exporters import JSONLExporter

    exporter = JSONLExporter(traces_dir)
    trace = exporter.load(trace_id)
    if trace is None:
        raise VincioError(f"trace {trace_id!r} not found in {exporter.path}")
    return trace


# -- commands -----------------------------------------------------------------------


def cmd_init(args: argparse.Namespace) -> int:
    root = Path(args.path)
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "vincio.yaml"
    if config_path.exists() and not args.force:
        return _fail(f"{config_path} already exists (use --force to overwrite)")
    config_path.write_text(
        yaml.safe_dump(
            {
                "project": args.project or root.resolve().name,
                "provider": {"default": "openai", "model": "gpt-5.2"},
                "storage": {"metadata": "sqlite:///.vincio/vincio.db"},
                "observability": {"exporter": "jsonl", "traces_dir": ".vincio/traces"},
                "security": {"tenant_isolation": True},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    app_path = root / "app.py"
    if not app_path.exists():
        app_path.write_text(
            '"""Vincio starter app."""\n\n'
            "from vincio import ContextApp\n\n"
            'app = ContextApp(name="my_app")\n'
            '# app.add_source("docs", path="./docs", retrieval="hybrid")\n'
            '# app.set_policy("answer_only_from_sources", True)\n\n'
            'if __name__ == "__main__":\n'
            '    result = app.run("Hello, Vincio!")\n'
            "    print(result.output)\n",
            encoding="utf-8",
        )
    golden = root / "golden"
    golden.mkdir(exist_ok=True)
    sample = golden / "basic.jsonl"
    if not sample.exists():
        sample.write_text(
            json.dumps(
                {
                    "id": "case_001",
                    "input": "What does this project do?",
                    "expected": "It answers questions over the project documents.",
                    "tags": ["smoke"],
                    "difficulty": "easy",
                }
            )
            + "\n",
            encoding="utf-8",
        )
    (root / ".vincio").mkdir(exist_ok=True)
    print(f"initialized vincio project in {root.resolve()}")
    print("  vincio.yaml      project configuration")
    print("  app.py           starter ContextApp")
    print("  golden/basic.jsonl  starter eval dataset")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    app = _load_app(args.app)
    result = app.run(
        args.input,
        files=args.file or None,
        tenant_id=args.tenant,
        user_id=args.user,
    )
    print(f"status: {result.status.value}")
    print(f"trace:  {result.trace_id}")
    if result.cost_usd:
        print(f"cost:   ${result.cost_usd:.6f}")
    if result.error:
        print(f"error:  {result.error}")
    print("output:")
    output = result.output
    if hasattr(output, "model_dump"):
        print(json_dumps(output.model_dump(), indent=2))
    elif isinstance(output, (dict, list)):
        print(json_dumps(output, indent=2))
    else:
        print(output)
    return 0 if result.error is None else 1


def cmd_eval_run(args: argparse.Namespace) -> int:
    from ..evals.reports import EvalReport
    from ..evals.runners import EvalRunner

    app = _load_app(args.app)
    gates: dict[str, str] = {}
    for gate in args.gate or []:
        if "=" not in gate:
            return _fail(f"invalid gate {gate!r}; use metric='>= 0.9'")
        key, _, expression = gate.partition("=")
        gates[key.strip()] = expression.strip()
    runner = EvalRunner(
        app,
        metrics=args.metric or None,
        concurrency=args.concurrency,
        gates=gates or None,
    )
    baseline = EvalReport.load(args.compare) if args.compare else None
    report = runner.run(args.dataset, baseline=baseline)
    report.print_summary()
    if args.output:
        report.save(args.output)
        print(f"\nsaved report to {args.output}")
    if baseline is not None:
        diff = report.metadata.get("baseline_diff", {})
        regressions = diff.get("regressed_cases", [])
        print(f"\nbaseline diff: {len(regressions)} regressed case-metric(s)")
        for regression in regressions[:10]:
            print(f"  - {regression['case_id']}: {regression['metric']} {regression['from']} → {regression['to']}")
    failed_gates = [k for k, v in report.gates.items() if not v.get("passed")]
    return 1 if failed_gates else 0


def cmd_eval_report(args: argparse.Namespace) -> int:
    from ..evals.reports import EvalReport

    path = Path(args.report)
    if path.is_dir():
        candidates = sorted(path.glob("*.json"))
        if not candidates:
            return _fail(f"no reports in {path}")
        path = candidates[-1]
    EvalReport.load(path).print_summary()
    return 0


def _spec_from_file(path: Path):
    from ..prompts.templates import PromptSpec

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise VincioError(f"{path}: prompt spec must be a mapping")
    data.setdefault("name", path.stem)
    return PromptSpec.model_validate(data)


def cmd_prompt_lint(args: argparse.Namespace) -> int:
    from ..prompts.lint import lint_spec

    target = Path(args.path)
    files = (
        sorted([*target.glob("*.yaml"), *target.glob("*.yml"), *target.glob("*.json")])
        if target.is_dir()
        else [target]
    )
    if not files:
        return _fail(f"no prompt files found in {target}")
    error_count = 0
    for file in files:
        try:
            spec = _spec_from_file(file)
        except (VincioError, yaml.YAMLError, ValueError) as exc:
            print(f"{file}: parse error: {exc}")
            error_count += 1
            continue
        findings = lint_spec(spec)
        if not findings:
            print(f"{file}: ok")
        for finding in findings:
            print(f"{file}: {finding.severity.upper()} {finding.code} [{finding.location}] {finding.message}")
            if finding.hint:
                print(f"    hint: {finding.hint}")
            if finding.severity == "error":
                error_count += 1
    return 1 if error_count else 0


def cmd_prompt_compile(args: argparse.Namespace) -> int:
    from ..prompts.compiler import CompilerOptions, PromptCompiler

    spec = _spec_from_file(Path(args.path))
    compiler = PromptCompiler(CompilerOptions(format=args.format))
    compiled = compiler.compile(spec, user_task=args.task or "")
    print(f"# prompt_id: {compiled.prompt_id}")
    print(f"# spec_hash: {compiled.prompt_spec_hash}  rendered_hash: {compiled.rendered_hash}")
    print(f"# tokens: {compiled.token_count}  cacheability: {compiled.cacheability:.2%}")
    for finding in compiled.lint_findings:
        print(f"# lint {finding.severity}: {finding.code} {finding.message}")
    print()
    for message in compiled.messages:
        print(f"--- {message.role}{' (cached prefix)' if message.cache_hint else ''} ---")
        print(message.text)
    return 0


def cmd_trace_show(args: argparse.Namespace) -> int:
    trace = _load_trace(args.trace_id, args.traces_dir)
    print(f"trace {trace.id}  app={trace.app_name}  status={trace.status}  duration={trace.duration_ms}ms")
    if trace.attributes:
        print(f"attributes: {json_dumps(trace.attributes)}")

    def render(nodes: list[dict[str, Any]], depth: int = 0) -> None:
        for node in nodes:
            indent = "  " * depth
            print(f"{indent}├─ [{node['type']}] {node['name']}  {node['status']}  {node['duration_ms']}ms")
            interesting = {
                k: v for k, v in node["attributes"].items()
                if k in ("model", "tokens", "cacheability", "evidence", "cost_usd", "valid", "task_type")
            }
            if interesting:
                print(f"{indent}│    {json_dumps(interesting)}")
            render(node["children"], depth + 1)

    render(trace.span_tree())
    return 0


def cmd_trace_replay(args: argparse.Namespace) -> int:
    from ..observability.traces import trace_replay_plan

    trace = _load_trace(args.trace_id, args.traces_dir)
    print(json_dumps(trace_replay_plan(trace), indent=2))
    return 0


def cmd_trace_diff(args: argparse.Namespace) -> int:
    from ..observability.traces import trace_diff

    trace_a = _load_trace(args.trace_a, args.traces_dir)
    trace_b = _load_trace(args.trace_b, args.traces_dir)
    print(json_dumps(trace_diff(trace_a, trace_b), indent=2))
    return 0


def cmd_optimize_run(args: argparse.Namespace) -> int:
    from ..evals.datasets import Dataset
    from ..evals.runners import EvalRunner
    from ..optimize.prompt_search import PromptOptimizer
    from ..optimize.search import FitnessWeights
    from ..providers.base import run_sync

    app = _load_app(args.app)
    dataset = Dataset.load(args.dataset)
    metrics = ["semantic_similarity", "schema_validity", "groundedness", "cost", "latency"]
    weights = FitnessWeights()
    if args.target == "groundedness":
        weights.groundedness = 2.0
    elif args.target == "cost":
        weights.cost = 2.0
    elif args.target == "latency":
        weights.latency = 1.0
    elif args.target == "quality":
        weights.accuracy = 2.0

    async def evaluate_variant(variant, ds):
        original_spec = app.prompt_spec
        original_options = app.prompt_compiler.options
        app.prompt_spec = variant.spec
        app.prompt_compiler.options = variant.compiler_options
        try:
            runner = EvalRunner(app, metrics=metrics, concurrency=args.concurrency)
            return await runner.arun(ds)
        finally:
            app.prompt_spec = original_spec
            app.prompt_compiler.options = original_options

    optimizer = PromptOptimizer(evaluate_variant, weights=weights)
    result = run_sync(
        optimizer.optimize(app.prompt_spec, dataset, max_variants=args.budget, subset_size=args.subset)
    )
    print(f"baseline fitness: {result.baseline_fitness:.4f}")
    for entry in result.history:
        print(f"  [{entry['phase']}] {entry['name']}: {entry['fitness']:.4f}")
    print(f"promoted: {result.promoted} — {result.reason}")
    if result.promoted and result.best is not None and args.output:
        Path(args.output).write_text(
            yaml.safe_dump(result.best.payload.spec.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )
        print(f"wrote winning prompt spec to {args.output}")
    return 0


def cmd_index_build(args: argparse.Namespace) -> int:
    from ..core.types import Document
    from ..documents.loaders import load_directory, load_document
    from ..retrieval.chunking import chunk_document
    from ..storage.sqlite import SQLiteMetadataStore

    target = Path(args.path)
    documents: list[Document] = []
    if target.is_dir():
        documents = load_directory(target)
    elif target.is_file():
        documents = [load_document(target)]
    else:
        return _fail(f"path not found: {target}")
    store = SQLiteMetadataStore(args.db)
    total_chunks = 0
    for document in documents:
        chunks = chunk_document(document, strategy=args.chunking, size=args.chunk_size)
        total_chunks += len(chunks)
        store.save(
            "documents",
            {"id": document.id, "title": document.title, "uri": document.source_uri, "media_type": document.media_type},
        )
        for chunk in chunks:
            store.save("chunks", chunk.model_dump(mode="json"))
    print(f"indexed {len(documents)} document(s), {total_chunks} chunk(s) into {args.db}")
    return 0


def _memory_engine(db: str):
    from ..memory.engine import MemoryEngine
    from ..memory.stores import SQLiteMemoryStore
    from ..retrieval.embeddings import LocalHashEmbedder
    from ..security.audit import AuditLog

    return MemoryEngine(
        SQLiteMemoryStore(db),
        embedder=LocalHashEmbedder(),
        audit=AuditLog(Path(db).parent / "audit"),
    )


def cmd_memory_remember(args: argparse.Namespace) -> int:
    engine = _memory_engine(args.db)
    item = engine.remember(
        args.content,
        user_id=args.user,
        agent_id=args.agent,
        session_id=args.session,
        tenant_id=args.tenant,
        scope=args.scope,
        type=args.type,
    )
    print(f"{item.id}  [{item.scope.value}/{item.type.value}]  conf={item.confidence:.2f}")
    return 0


def cmd_memory_recall(args: argparse.Namespace) -> int:
    engine = _memory_engine(args.db)
    results = engine.search(
        args.query,
        user_id=args.user,
        agent_id=args.agent,
        session_id=args.session,
        tenant_id=args.tenant,
        top_k=args.top_k,
    )
    if not results:
        print("no memories recalled")
        return 0
    for result in results:
        print(f"{result.score:8.4f}  {result.item.id}  [{result.item.scope.value}/{result.item.type.value}]")
        print(f"          {result.item.content[:140]}")
    return 0


def cmd_memory_forget(args: argparse.Namespace) -> int:
    engine = _memory_engine(args.db)
    if engine.forget(args.memory_id, reason=args.reason):
        print(f"forgot {args.memory_id}")
        return 0
    return _fail(f"memory not found: {args.memory_id}")


def cmd_memory_export(args: argparse.Namespace) -> int:
    engine = _memory_engine(args.db)
    records = engine.export_owner_data(args.owner)
    output = json_dumps(records)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"exported {len(records)} memorie(s) to {args.output}")
    else:
        print(output)
    return 0


def cmd_memory_consolidate(args: argparse.Namespace) -> int:
    import asyncio

    engine = _memory_engine(args.db)
    report = asyncio.run(engine.consolidate(args.session_id, user_id=args.user))
    print(
        f"examined={report.examined} promoted={report.promoted} "
        f"deduplicated={report.deduplicated} archived={report.archived}"
    )
    return 0


def cmd_memory_decay(args: argparse.Namespace) -> int:
    engine = _memory_engine(args.db)
    stats = engine.decay_pass()
    print(f"decayed={stats['decayed']} archived={stats['archived']} expired={stats['expired']}")
    return 0


def cmd_memory_inspect(args: argparse.Namespace) -> int:
    from ..memory.stores import SQLiteMemoryStore

    store = SQLiteMemoryStore(args.db)
    items = store.all_items(owner_id=args.user, statuses=())
    if args.user:
        items = [i for i in items if i.owner_id == args.user]
    if not items:
        print("no memories found")
        return 0
    for item in sorted(items, key=lambda i: i.updated_at, reverse=True)[: args.limit]:
        print(
            f"{item.id}  [{item.scope.value}/{item.type.value}]  conf={item.confidence:.2f}  "
            f"status={item.status}  owner={item.owner_id or '-'}"
        )
        print(f"   {item.content[:140]}")
    return 0


# -- parser ----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vincio", description="Vincio context engineering platform")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="initialize a vincio project")
    p_init.add_argument("path", nargs="?", default=".")
    p_init.add_argument("--project", default=None)
    p_init.add_argument("--force", action="store_true")
    p_init.set_defaults(fn=cmd_init)

    p_run = sub.add_parser("run", help="run an app once")
    p_run.add_argument("app", help="python file exposing a ContextApp as `app`")
    p_run.add_argument("--input", required=True)
    p_run.add_argument("--file", action="append", help="attach a file (repeatable)")
    p_run.add_argument("--tenant", default=None)
    p_run.add_argument("--user", default=None)
    p_run.set_defaults(fn=cmd_run)

    p_eval = sub.add_parser("eval", help="evaluation commands")
    eval_sub = p_eval.add_subparsers(dest="eval_command", required=True)
    p_eval_run = eval_sub.add_parser("run", help="run an eval dataset")
    p_eval_run.add_argument("dataset")
    p_eval_run.add_argument("--app", required=True)
    p_eval_run.add_argument("--metric", action="append", help="metric name (repeatable)")
    p_eval_run.add_argument("--gate", action="append", help="gate, e.g. groundedness='>= 0.9'")
    p_eval_run.add_argument("--compare", default=None, help="baseline report JSON to diff against")
    p_eval_run.add_argument("--output", default=None, help="save report JSON here")
    p_eval_run.add_argument("--concurrency", type=int, default=8)
    p_eval_run.set_defaults(fn=cmd_eval_run)
    p_eval_report = eval_sub.add_parser("report", help="print a saved report")
    p_eval_report.add_argument("report", help="report JSON file or directory")
    p_eval_report.set_defaults(fn=cmd_eval_report)

    p_prompt = sub.add_parser("prompt", help="prompt tooling")
    prompt_sub = p_prompt.add_subparsers(dest="prompt_command", required=True)
    p_lint = prompt_sub.add_parser("lint", help="lint prompt spec files")
    p_lint.add_argument("path")
    p_lint.set_defaults(fn=cmd_prompt_lint)
    p_compile = prompt_sub.add_parser("compile", help="compile a prompt spec")
    p_compile.add_argument("path")
    p_compile.add_argument("--format", default="markdown", choices=["markdown", "xml", "json", "minimal"])
    p_compile.add_argument("--task", default=None, help="user task to render")
    p_compile.set_defaults(fn=cmd_prompt_compile)

    p_trace = sub.add_parser("trace", help="trace tooling")
    trace_sub = p_trace.add_subparsers(dest="trace_command", required=True)
    for name, fn, extra in (
        ("show", cmd_trace_show, ["trace_id"]),
        ("replay", cmd_trace_replay, ["trace_id"]),
        ("diff", cmd_trace_diff, ["trace_a", "trace_b"]),
    ):
        p_trace_sub = trace_sub.add_parser(name)
        for argument in extra:
            p_trace_sub.add_argument(argument)
        p_trace_sub.add_argument("--traces-dir", default=".vincio/traces")
        p_trace_sub.set_defaults(fn=fn)

    p_optimize = sub.add_parser("optimize", help="optimization commands")
    optimize_sub = p_optimize.add_subparsers(dest="optimize_command", required=True)
    p_opt_run = optimize_sub.add_parser("run", help="optimize the app prompt against a dataset")
    p_opt_run.add_argument("--app", required=True)
    p_opt_run.add_argument("--dataset", required=True)
    p_opt_run.add_argument("--target", default="quality", choices=["quality", "groundedness", "cost", "latency"])
    p_opt_run.add_argument("--budget", type=int, default=8, help="max prompt variants")
    p_opt_run.add_argument("--subset", type=int, default=8, help="screening subset size")
    p_opt_run.add_argument("--concurrency", type=int, default=4)
    p_opt_run.add_argument("--output", default=None, help="write winning spec YAML here")
    p_opt_run.set_defaults(fn=cmd_optimize_run)

    p_index = sub.add_parser("index", help="index commands")
    index_sub = p_index.add_subparsers(dest="index_command", required=True)
    p_index_build = index_sub.add_parser("build", help="load, chunk, and persist documents")
    p_index_build.add_argument("path")
    p_index_build.add_argument("--db", default=".vincio/vincio.db")
    p_index_build.add_argument("--chunking", default="adaptive")
    p_index_build.add_argument("--chunk-size", type=int, default=400)
    p_index_build.set_defaults(fn=cmd_index_build)

    p_memory = sub.add_parser("memory", help="memory commands")
    memory_sub = p_memory.add_subparsers(dest="memory_command", required=True)
    p_mem_inspect = memory_sub.add_parser("inspect", help="list stored memories")
    p_mem_inspect.add_argument("--user", default=None)
    p_mem_inspect.add_argument("--db", default=".vincio/memory.db")
    p_mem_inspect.add_argument("--limit", type=int, default=50)
    p_mem_inspect.set_defaults(fn=cmd_memory_inspect)

    def _owner_flags(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--db", default=".vincio/memory.db")
        parser.add_argument("--user", default=None)
        parser.add_argument("--agent", default=None)
        parser.add_argument("--session", default=None)
        parser.add_argument("--tenant", default=None)

    p_mem_remember = memory_sub.add_parser("remember", help="write one memory")
    p_mem_remember.add_argument("content")
    _owner_flags(p_mem_remember)
    p_mem_remember.add_argument("--scope", default=None)
    p_mem_remember.add_argument("--type", default=None)
    p_mem_remember.set_defaults(fn=cmd_memory_remember)

    p_mem_recall = memory_sub.add_parser("recall", help="scored hybrid recall")
    p_mem_recall.add_argument("query")
    _owner_flags(p_mem_recall)
    p_mem_recall.add_argument("--top-k", type=int, default=5, dest="top_k")
    p_mem_recall.set_defaults(fn=cmd_memory_recall)

    p_mem_forget = memory_sub.add_parser("forget", help="delete one memory (audited)")
    p_mem_forget.add_argument("memory_id")
    p_mem_forget.add_argument("--db", default=".vincio/memory.db")
    p_mem_forget.add_argument("--reason", default="user_request")
    p_mem_forget.set_defaults(fn=cmd_memory_forget)

    p_mem_export = memory_sub.add_parser("export", help="GDPR-style owner export (audited)")
    p_mem_export.add_argument("--owner", required=True)
    p_mem_export.add_argument("--db", default=".vincio/memory.db")
    p_mem_export.add_argument("--output", default=None)
    p_mem_export.set_defaults(fn=cmd_memory_export)

    p_mem_consolidate = memory_sub.add_parser(
        "consolidate", help="episodic→semantic consolidation for a session"
    )
    p_mem_consolidate.add_argument("session_id")
    p_mem_consolidate.add_argument("--db", default=".vincio/memory.db")
    p_mem_consolidate.add_argument("--user", default=None)
    p_mem_consolidate.set_defaults(fn=cmd_memory_consolidate)

    p_mem_decay = memory_sub.add_parser("decay", help="run a decay/TTL pass")
    p_mem_decay.add_argument("--db", default=".vincio/memory.db")
    p_mem_decay.set_defaults(fn=cmd_memory_decay)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.fn(args)
    except VincioError as exc:
        return _fail(exc.message)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
