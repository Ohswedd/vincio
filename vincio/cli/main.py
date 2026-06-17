"""Vincio CLI.

Commands::

    vincio init --template rag|agent|eval
    vincio config schema --output vincio.schema.json
    vincio config validate vincio.yaml
    vincio config show
    vincio packs list
    vincio packs show support
    vincio tui
    vincio run app.py --input "..."
    vincio batch app.py --input "..." --input "..."   # Batch API, ~50% cost
    vincio cost report --by tenant|feature             # attributed cost rollup
    vincio eval run golden.jsonl --app app.py
    vincio eval report <report.json>
    vincio eval dataset golden.jsonl [--group-by-session]
    vincio eval drift baseline.json current.json [--threshold 0.1]
    vincio eval annotate labels.jsonl [--threshold 0.6] [--bins 2]
    vincio prompt lint prompts/
    vincio prompt compile prompt.yaml
    vincio trace show <trace_id>
    vincio trace replay <trace_id>
    vincio trace diff <trace_a> <trace_b>
    vincio optimize run --app app.py --dataset golden.jsonl --target groundedness
    vincio loop run --app app.py --min-feedback 0.5 --gate groundedness=">= 0.8"
    vincio index build ./docs
    vincio memory inspect --user u1
    vincio memory remember "Prefers concise answers" --user u1
    vincio memory recall "answer style" --user u1
    vincio memory forget <memory_id>
    vincio memory export --owner u1
    vincio memory consolidate <session_id> --user u1
    vincio memory decay
    vincio audit verify .vincio/audit/audit.jsonl
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


def _golden_line(**case: Any) -> str:
    return json.dumps(case) + "\n"


def _vincio_yaml(project: str, provider: str, **sections: Any) -> str:
    config: dict[str, Any] = {
        "project": project,
        "provider": {"default": provider, "model": "gpt-5.2"},
        "storage": {"metadata": "sqlite:///.vincio/vincio.db"},
        "observability": {"exporter": "jsonl", "traces_dir": ".vincio/traces"},
        "security": {"tenant_isolation": True},
    }
    for key, value in sections.items():
        config[key] = {**config.get(key, {}), **value} if isinstance(value, dict) else value
    body = yaml.safe_dump(config, sort_keys=False)
    return "# yaml-language-server: $schema=./vincio.schema.json\n" + body


def _template_files(template: str, project: str, provider: str) -> dict[str, str]:
    """Return ``relative_path -> contents`` for a scaffold template."""
    if template == "rag":
        return {
            "vincio.yaml": _vincio_yaml(
                project, provider, retrieval={"reranker": "heuristic", "embedder": "local"}
            ),
            "app.py": (
                '"""Vincio RAG app: answer questions grounded in ./docs."""\n\n'
                "from vincio import ContextApp\n\n"
                f'app = ContextApp(name="{project}")\n'
                'app.add_source("docs", path="./docs", retrieval="hybrid")\n'
                'app.set_policy("answer_only_from_sources", True)\n\n'
                'if __name__ == "__main__":\n'
                '    result = app.run("What does this project do?")\n'
                "    print(result.output)\n"
                "    print(result.citations)\n"
            ),
            "docs/welcome.md": (
                "# Welcome\n\nThis project answers questions over the documents in `./docs`.\n"
                "Add your own Markdown, PDF, or text files here and re-run.\n"
            ),
            "golden/qa.jsonl": (
                _golden_line(
                    id="qa_001",
                    input="What does this project do?",
                    expected="It answers questions grounded in the project documents.",
                    rubric={"answer_only_from_sources": True},
                    tags=["smoke"],
                    difficulty="easy",
                )
            ),
        }
    if template == "agent":
        return {
            "vincio.yaml": _vincio_yaml(project, provider),
            "app.py": (
                '"""Vincio agent app: a ContextApp with a tool."""\n\n'
                "from vincio import ContextApp\n\n\n"
                "def get_weather(city: str) -> dict:\n"
                '    """Look up the current weather for a city."""\n'
                '    return {"city": city, "conditions": "sunny", "temp_c": 22}\n\n\n'
                f'app = ContextApp(name="{project}")\n'
                'app.add_tool(get_weather, permission="read_only")\n\n'
                'if __name__ == "__main__":\n'
                '    result = app.run("What is the weather in Rome?")\n'
                "    print(result.output)\n"
            ),
            "golden/agent.jsonl": _golden_line(
                id="agent_001",
                input="What is the weather in Rome?",
                expected="It is sunny and 22C in Rome.",
                tags=["tool"],
                difficulty="easy",
            ),
        }
    if template == "eval":
        cases = [
            _golden_line(id="eval_001", input="2 + 2", expected="4", tags=["math"], difficulty="easy"),
            _golden_line(
                id="eval_002",
                input="Summarize: the cat sat on the mat.",
                expected="A cat sat on a mat.",
                tags=["summary"],
                difficulty="medium",
            ),
        ]
        return {
            "vincio.yaml": _vincio_yaml(project, provider),
            "app.py": (
                '"""Vincio app under evaluation."""\n\n'
                "from vincio import ContextApp\n\n"
                f'app = ContextApp(name="{project}")\n\n'
                'if __name__ == "__main__":\n'
                '    print(app.run("2 + 2").output)\n'
            ),
            "golden/eval.jsonl": "".join(cases),
            "README.md": (
                f"# {project}\n\nRun the eval suite:\n\n"
                "```sh\nvincio eval run golden/eval.jsonl --app app.py \\\n"
                '  --metric semantic_similarity --gate "semantic_similarity=>= 0.6"\n```\n'
            ),
        }
    # minimal (default)
    return {
        "vincio.yaml": _vincio_yaml(project, provider),
        "app.py": (
            '"""Vincio starter app."""\n\n'
            "from vincio import ContextApp\n\n"
            f'app = ContextApp(name="{project}")\n'
            '# app.add_source("docs", path="./docs", retrieval="hybrid")\n'
            '# app.set_policy("answer_only_from_sources", True)\n'
            '# app.use_pack("support")  # opt-in domain bundle\n\n'
            'if __name__ == "__main__":\n'
            '    result = app.run("Hello, Vincio!")\n'
            "    print(result.output)\n"
        ),
        "golden/basic.jsonl": _golden_line(
            id="case_001",
            input="What does this project do?",
            expected="It answers questions over the project documents.",
            tags=["smoke"],
            difficulty="easy",
        ),
    }


def cmd_init(args: argparse.Namespace) -> int:
    from ..core.config import config_json_schema

    root = Path(args.path)
    root.mkdir(parents=True, exist_ok=True)
    project = args.project or root.resolve().name
    template = args.template
    files = _template_files(template, project, args.provider)
    # Always write a JSON Schema so the vincio.yaml $schema hint resolves.
    files["vincio.schema.json"] = json_dumps(config_json_schema(), indent=2) + "\n"

    config_path = root / "vincio.yaml"
    if config_path.exists() and not args.force:
        return _fail(f"{config_path} already exists (use --force to overwrite)")

    written: list[str] = []
    for relative, contents in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not args.force and relative != "vincio.yaml":
            continue
        path.write_text(contents, encoding="utf-8")
        written.append(relative)
    (root / ".vincio").mkdir(exist_ok=True)

    print(f"initialized vincio project ({template} template) in {root.resolve()}")
    for relative in sorted(written):
        print(f"  {relative}")
    print(f"\nnext: cd {root} && vincio run app.py --input \"Hello, Vincio!\"")
    return 0


def cmd_config_schema(args: argparse.Namespace) -> int:
    from ..core.config import config_json_schema

    payload = json_dumps(config_json_schema(), indent=2)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
        print(f"wrote JSON Schema to {args.output}")
    else:
        print(payload)
    return 0


def cmd_config_validate(args: argparse.Namespace) -> int:
    from ..core.config import find_config_file, load_config

    path = args.path or find_config_file()
    if path is None:
        return _fail("no vincio config file found")
    try:
        config = load_config(path)
    except VincioError as exc:
        return _fail(f"{path}: {exc.message}")
    print(f"{path}: ok (project={config.project}, provider={config.provider.default})")
    return 0


def cmd_config_show(args: argparse.Namespace) -> int:
    from ..core.config import load_config

    config = load_config(args.path) if args.path else load_config()
    print(config.to_yaml())
    return 0


def cmd_packs_list(args: argparse.Namespace) -> int:
    from ..packs import available_packs, load_pack

    for name in available_packs():
        try:
            pack = load_pack(name)
            print(f"{name:14s} {pack.description}")
        except VincioError:
            print(f"{name:14s} (failed to load)")
    return 0


def cmd_packs_show(args: argparse.Namespace) -> int:
    from ..packs import load_pack

    pack = load_pack(args.name)
    print(f"# pack: {pack.name}")
    print(f"description: {pack.description}")
    print(f"role:        {pack.role}")
    print(f"objective:   {pack.objective}")
    print(f"rules:       {len(pack.rules)}")
    print(f"policies:    {pack.policies}")
    print(f"evaluators:  {', '.join(pack.evaluators) or '-'}")
    print(f"eval cases:  {len(pack.eval_cases)}")
    if pack.output_schema:
        print(f"output schema ({pack.output_schema_name or pack.name}):")
        print(json_dumps(pack.output_schema, indent=2))
    return 0


def cmd_tui(args: argparse.Namespace) -> int:
    from ..tui import TUI

    TUI(traces_dir=args.traces_dir, memory_db=args.db).run()
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


def cmd_batch(args: argparse.Namespace) -> int:
    app = _load_app(args.app)
    inputs: list[str] = list(args.input or [])
    if args.input_file:
        text = Path(args.input_file).read_text(encoding="utf-8")
        inputs.extend(line for line in (ln.strip() for ln in text.splitlines()) if line)
    if not inputs:
        return _fail("no inputs; pass --input (repeatable) or --input-file")
    results = app.batch(inputs, discount=args.discount)
    succeeded = sum(1 for r in results if r.error is None)
    total_cost = sum(r.cost_usd for r in results)
    print(f"batch: {succeeded}/{len(results)} succeeded  cost=${total_cost:.6f}")
    for i, result in enumerate(results):
        marker = "ok " if result.error is None else "ERR"
        print(f"  [{marker}] {i}: {result.trace_id}  ${result.cost_usd:.6f}")
        if result.error:
            print(f"        {result.error}")
    if args.output:
        Path(args.output).write_text(
            json_dumps([r.model_dump(mode="json") for r in results], indent=2), encoding="utf-8"
        )
        print(f"saved results to {args.output}")
    return 0 if succeeded == len(results) else 1


def cmd_cost_report(args: argparse.Namespace) -> int:
    from ..observability.finops import CostLedger
    from ..storage.base import create_metadata_store

    store = create_metadata_store(f"sqlite:///{args.db}")
    ledger = CostLedger.from_store(store)
    report = ledger.report(args.by)
    if args.json:
        print(json_dumps(report.model_dump(mode="json"), indent=2))
    else:
        report.print_summary()
    return 0


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


def cmd_eval_regress(args: argparse.Namespace) -> int:
    from ..providers.base import run_sync

    app = _load_app(args.app)
    report = run_sync(
        app.aswap_regression(
            args.dataset,
            candidate_model=args.candidate_model,
            baseline_model=args.baseline_model,
            metrics=args.metric or None,
            quality_metric=args.quality_metric,
            alpha=args.alpha,
            repeats=args.repeats,
            flake_quarantine=not args.no_flake_quarantine,
        )
    )
    print(
        f"model-swap regression: {report.baseline_model} → {report.candidate_model} "
        f"({report.n_cases} cases)"
    )
    for metric, test in sorted(report.metric_tests.items()):
        flag = "REGRESSION" if metric in report.regressions else (
            "significant" if test["significant"] else "ns"
        )
        print(
            f"  {metric:24s} {test['mean_a']:.4f} → {test['mean_b']:.4f}  "
            f"Δ {test['delta']:+.4f}  p={test['p_value']:.4f}  [{flag}]"
        )
    if report.cost:
        print(
            f"  cost ${report.cost['baseline']:.6f} → ${report.cost['candidate']:.6f} "
            f"(×{report.cost['ratio']})  latency {report.latency['baseline']:.0f}ms → "
            f"{report.latency['candidate']:.0f}ms"
        )
    if report.worst_slices:
        print("  worst-regressed slices:")
        for s in report.worst_slices:
            print(f"    - {s['slice']}: {s['baseline']:.4f} → {s['candidate']:.4f} ({s['delta']:+.4f})")
    if report.flaky_excluded:
        print(f"  ({report.flaky_excluded} flaky case(s) quarantined from gates)")
    if args.output:
        Path(args.output).write_text(json_dumps(report.model_dump()), encoding="utf-8")
        print(f"saved to {args.output}")
    if report.regressed:
        print(f"\nREGRESSION: {', '.join(report.regressions)} significantly worse")
        return 1
    print("\nno significant quality regression — safe to swap")
    return 0


def cmd_providers_list(args: argparse.Namespace) -> int:
    from ..providers.registry import default_model_registry

    registry = default_model_registry()
    profiles = sorted(registry.profiles(), key=lambda p: (p.provider, p.model))
    rows = [
        {
            "model": p.model, "provider": p.provider, "tier": p.tier,
            "lifecycle": p.lifecycle(), "input_per_mtok": p.input_cost_per_mtok,
            "output_per_mtok": p.output_cost_per_mtok, "successor": p.successor,
        }
        for p in profiles
        if not args.provider or p.provider == args.provider
    ]
    if args.json:
        print(json_dumps(rows))
        return 0
    print(f"{'model':28s} {'provider':10s} {'tier':8s} {'lifecycle':10s} {'in/out $/Mtok':>16s}")
    for r in rows:
        print(
            f"{r['model']:28s} {r['provider']:10s} {r['tier']:8s} {r['lifecycle']:10s} "
            f"{r['input_per_mtok']:>7}/{r['output_per_mtok']:<8}"
        )
    print(f"\n{len(rows)} model(s)")
    return 0


def cmd_providers_lifecycle(args: argparse.Namespace) -> int:
    from datetime import date

    from ..providers.lifecycle import LifecycleWatcher

    as_of = date.fromisoformat(args.as_of) if args.as_of else None
    if args.app:
        app = _load_app(args.app)
        result = app.watch_lifecycle(
            args.model or None, as_of=as_of, warn_within_days=args.warn_within_days
        )
        alerts, proposals = result["alerts"], result["proposals"]
    else:
        if not args.model:
            return _fail("provide --model (repeatable) or --app")
        watcher = LifecycleWatcher(warn_within_days=args.warn_within_days)
        alerts = watcher.scan(args.model, as_of=as_of)
        proposals = watcher.propose_all(args.model, as_of=as_of)
    if args.json:
        print(json_dumps({
            "alerts": [a.model_dump() for a in alerts],
            "proposals": [p.model_dump() for p in proposals],
        }))
    else:
        for alert in alerts:
            print(f"[{alert.severity.upper():8s}] {alert.message}")
        for proposal in proposals:
            print(
                f"  → migrate {proposal.from_model} to {proposal.to_model} ({proposal.kind}; "
                f"{proposal.savings_pct:+.0%} blended cost, "
                f"{'capability superset' if proposal.capability_superset else 'check capabilities'})"
            )
        if not alerts:
            print("no models nearing sunset")
    return 1 if any(a.severity in ("warn", "critical") for a in alerts) else 0


def cmd_providers_discover(args: argparse.Namespace) -> int:
    from ..providers import build_provider
    from ..providers.base import run_sync
    from ..providers.discovery import discover_models

    provider = build_provider(args.provider, with_retries=False)
    summary = run_sync(
        discover_models(provider, mark_missing_deprecated=args.mark_missing_deprecated)
    )
    if args.json:
        print(json_dumps(summary))
    else:
        print(
            f"discovered against {args.provider}: {len(summary['added'])} added, "
            f"{len(summary['updated'])} updated, "
            f"{len(summary['deprecated_missing'])} flagged deprecated"
        )
        for key in ("added", "updated", "deprecated_missing"):
            if summary[key]:
                print(f"  {key}: {', '.join(summary[key])}")
    return 0


def cmd_providers_regress(args: argparse.Namespace) -> int:
    from ..providers.base import run_sync

    app = _load_app(args.app)
    gates: dict[str, str] = {}
    for gate in args.gate or []:
        if "=" not in gate:
            return _fail(f"invalid gate {gate!r}; use metric='>= 0.9'")
        key, _, expression = gate.partition("=")
        gates[key.strip()] = expression.strip()
    traces = [_load_trace(t, args.traces_dir) for t in (args.trace or [])]
    verdict = run_sync(
        app.agate_swap(
            args.candidate_model,
            baseline_model=args.baseline_model,
            dataset=args.dataset,
            traces=traces or None,
            gates=gates or None,
            quality_metric=args.quality_metric,
            alpha=args.alpha,
            repeats=args.repeats,
        )
    )
    print(f"swap gate: {verdict.baseline_model} → {verdict.candidate_model}")
    print(f"  verdict: {'PASS' if verdict.passed else 'FAIL'} — {verdict.reason}")
    if verdict.replay is not None:
        print(f"  replay: {json_dumps(verdict.replay)}")
    print(json_dumps(verdict.summary()))
    return 0 if verdict.passed else 1


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


def cmd_prompt_push(args: argparse.Namespace) -> int:
    from ..prompts.registry import PromptRegistry

    registry = PromptRegistry(args.registry)
    spec = _spec_from_file(Path(args.path))
    version = registry.push(
        spec,
        name=args.name,
        tags=args.tag or None,
        message=args.message or "",
    )
    tags = f"  tags={','.join(version.tags)}" if version.tags else ""
    print(f"{version.ref}  hash={version.spec_hash[:12]}{tags}")
    return 0


def cmd_prompt_versions(args: argparse.Namespace) -> int:
    from ..prompts.registry import PromptRegistry

    registry = PromptRegistry(args.registry)
    for version in registry.versions(args.name):
        tags = f"  [{', '.join(version.tags)}]" if version.tags else ""
        message = f"  {version.message}" if version.message else ""
        evals = f"  evals={len(version.eval_runs)}" if version.eval_runs else ""
        print(f"v{version.version}  {version.spec_hash[:12]}{tags}{message}{evals}")
    return 0


def cmd_prompt_diff(args: argparse.Namespace) -> int:
    from ..prompts.registry import PromptRegistry

    registry = PromptRegistry(args.registry)
    result = registry.diff(args.name, args.version_a, args.version_b, rendered=args.rendered)
    if args.rendered and result.get("rendered_diff"):
        print(result.pop("rendered_diff"))
        print()
    print(json_dumps(result, indent=2))
    return 0


def cmd_prompt_rollback(args: argparse.Namespace) -> int:
    from ..prompts.registry import PromptRegistry

    registry = PromptRegistry(args.registry)
    version = registry.rollback(args.name, to_version=args.to)
    print(f"{version.ref}  {version.message}")
    return 0


def cmd_trace_show(args: argparse.Namespace) -> int:
    from ..observability.viewer import _INTERESTING_ATTRIBUTES

    trace = _load_trace(args.trace_id, args.traces_dir)
    print(f"trace {trace.id}  app={trace.app_name}  status={trace.status}  duration={trace.duration_ms}ms")
    if trace.attributes:
        print(f"attributes: {json_dumps(trace.attributes)}")

    def render(nodes: list[dict[str, Any]], depth: int = 0) -> None:
        for node in nodes:
            indent = "  " * depth
            print(f"{indent}├─ [{node['type']}] {node['name']}  {node['status']}  {node['duration_ms']}ms")
            interesting = {
                k: v for k, v in node["attributes"].items() if k in _INTERESTING_ATTRIBUTES
            }
            if interesting:
                print(f"{indent}│    {json_dumps(interesting)}")
            render(node["children"], depth + 1)

    render(trace.span_tree())
    return 0


def cmd_trace_replay(args: argparse.Namespace) -> int:
    from ..observability.traces import trace_replay_plan

    trace = _load_trace(args.trace_id, args.traces_dir)
    target = getattr(args, "against", None)
    if not target:
        # No target: print the extracted replay plan (unchanged behavior).
        print(json_dumps(trace_replay_plan(trace), indent=2))
        return 0

    # Execute the plan against a target app and diff outputs/trajectory/cost.
    from ..evals.replay import ReplayRunner
    from ..providers.base import run_sync

    app = _load_app(target)
    runner = ReplayRunner(app)
    result = run_sync(runner.replay([trace], pin_tools=getattr(args, "pin_tools", False)))
    print(json_dumps({"summary": result.summary(), "report_diff": result.report_diff,
                      "cases": [c.model_dump() for c in result.cases]}, indent=2))
    return 0


def cmd_trace_diff(args: argparse.Namespace) -> int:
    from ..observability.traces import trace_diff

    trace_a = _load_trace(args.trace_a, args.traces_dir)
    trace_b = _load_trace(args.trace_b, args.traces_dir)
    if getattr(args, "html", None):
        from ..observability.viewer import trace_diff_html

        Path(args.html).write_text(trace_diff_html(trace_a, trace_b), encoding="utf-8")
        print(f"wrote {args.html}")
        return 0
    print(json_dumps(trace_diff(trace_a, trace_b), indent=2))
    return 0


def cmd_trace_view(args: argparse.Namespace) -> int:
    from ..observability.viewer import render_trace_text

    trace = _load_trace(args.trace_id, args.traces_dir)
    print(render_trace_text(trace))
    return 0


def cmd_trace_export(args: argparse.Namespace) -> int:
    from ..observability.exporters import JSONLExporter
    from ..observability.sessions import sessions_from_traces
    from ..observability.viewer import session_to_html, trace_to_html

    if args.session:
        exporter = JSONLExporter(args.traces_dir)
        sessions = [s for s in sessions_from_traces(exporter.load_all()) if s.id == args.trace_id]
        if not sessions:
            raise VincioError(f"session {args.trace_id!r} not found in {exporter.path}")
        html_text = session_to_html(sessions[0])
    else:
        html_text = trace_to_html(_load_trace(args.trace_id, args.traces_dir))
    output = args.output or f"{args.trace_id}.html"
    Path(output).write_text(html_text, encoding="utf-8")
    print(f"wrote {output}")
    return 0


def cmd_trace_sessions(args: argparse.Namespace) -> int:
    from ..observability.exporters import JSONLExporter
    from ..observability.sessions import sessions_from_traces

    exporter = JSONLExporter(args.traces_dir)
    sessions = sessions_from_traces(exporter.load_all())
    if not sessions:
        print(f"no sessions found in {exporter.path}")
        return 0
    for session in sessions:
        print(json_dumps(session.summary()))
    return 0


def cmd_trace_feedback(args: argparse.Namespace) -> int:
    from ..observability.exporters import JSONLExporter
    from ..observability.sessions import record_feedback

    trace = _load_trace(args.trace_id, args.traces_dir)
    record_feedback(
        trace,
        key=args.key,
        score=args.score,
        comment=args.comment or "",
        user_id=args.user,
        exporter=JSONLExporter(args.traces_dir),
    )
    print(f"recorded feedback on {trace.id} (key={args.key}, score={args.score})")
    return 0


def cmd_eval_dataset(args: argparse.Namespace) -> int:
    from ..evals.datasets import dataset_from_traces
    from ..observability.exporters import JSONLExporter

    exporter = JSONLExporter(args.traces_dir)
    traces = exporter.load_all()
    dataset = dataset_from_traces(
        traces,
        name=args.name or Path(args.output).stem,
        min_feedback_score=args.min_feedback,
        group_by_session=getattr(args, "group_by_session", False),
    )
    dataset.save(args.output)
    print(f"wrote {len(dataset)} case(s) from {len(traces)} trace(s) to {args.output}")
    return 0


def cmd_eval_drift(args: argparse.Namespace) -> int:
    from ..evals.drift import DriftMonitor
    from ..evals.reports import EvalReport

    baseline = EvalReport.load(args.baseline)
    current = EvalReport.load(args.current)
    monitor = DriftMonitor(score_threshold=args.threshold)
    shared = sorted(set(baseline.summary()) & set(current.summary()))
    if args.metric:
        shared = [m for m in shared if m in args.metric]
    if not shared:
        return _fail("no shared metrics between the two reports")
    print(f"drift: baseline `{baseline.name}` vs `{current.name}` (threshold {args.threshold})")
    drifted: list[dict[str, Any]] = []
    for metric in shared:
        monitor.set_score_baseline(metric, baseline.metric_values(metric))
        report = monitor.check_scores(metric, current.metric_values(metric))
        flag = "DRIFT" if report.drifted else "ok"
        print(
            f"  [{flag:5s}] {metric:22s} baseline={report.baseline:.4f} "
            f"current={report.current:.4f} delta={report.delta:+.4f}"
        )
        if report.drifted:
            drifted.append(report.model_dump())
    if args.output:
        Path(args.output).write_text(json_dumps(drifted), encoding="utf-8")
        print(f"\nsaved {len(drifted)} drift record(s) to {args.output}")
    print(f"\n{len(drifted)} metric(s) drifted")
    return 1 if drifted else 0


def cmd_eval_annotate(args: argparse.Namespace) -> int:
    from ..evals.annotation import cohens_kappa

    pairs: list[tuple[float, float]] = []
    for line in Path(args.labels).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        judge = record.get("judge", record.get("judge_score"))
        human = record.get("human", record.get("human_score"))
        if judge is None or human is None:
            continue
        pairs.append((float(judge), float(human)))
    if len(pairs) < 2:
        return _fail("need at least 2 (judge, human) score pairs in the labels file")
    kappa = cohens_kappa(pairs, bins=args.bins)
    trusted = kappa >= args.threshold
    print(f"annotation agreement over {len(pairs)} pair(s):")
    print(f"  cohens_kappa = {kappa:.4f}  (threshold {args.threshold}, bins {args.bins})")
    print(f"  judge {'EARNS' if trusted else 'does NOT earn'} CI-gating weight")
    return 0 if trusted else 1


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


def cmd_loop_run(args: argparse.Namespace) -> int:
    """One improvement-loop cycle: trace → dataset → eval → optimize → promote."""
    from ..evals.datasets import Dataset
    from ..optimize.loop import ImprovementLoop
    from ..optimize.search import FitnessWeights

    app = _load_app(args.app)
    gates: dict[str, str] = {}
    for gate in args.gate or []:
        if "=" not in gate:
            return _fail(f"invalid gate {gate!r}; use metric='>= 0.9'")
        key, _, expression = gate.partition("=")
        gates[key.strip()] = expression.strip()
    loop = ImprovementLoop(
        app,
        weights=FitnessWeights(),
        gates=gates or None,
        experiment=args.experiment,
        concurrency=args.concurrency,
        optimizer="reflective" if getattr(args, "reflective", False) else "evolution",
    )
    dataset = Dataset.load(args.dataset) if args.dataset else None
    result = loop.run(
        dataset=dataset,
        min_feedback_score=args.min_feedback,
        max_variants=args.budget,
        subset_size=args.subset,
        promote_tag=args.tag,
        dry_run=args.dry_run,
    )
    print(f"dataset: {result.dataset_name} ({result.dataset_size} cases, fp {result.dataset_fingerprint})")
    for step in result.steps:
        detail = ", ".join(f"{k}={v}" for k, v in step.items() if k != "stage")
        print(f"  [{step['stage']}] {detail}")
    if result.optimization is not None:
        for entry in result.optimization.history:
            if entry["phase"] != "subset":
                fitness_value = entry.get("fitness")
                if isinstance(fitness_value, float):
                    print(f"  [{entry['phase']}] {entry['name']}: {fitness_value:.4f}")
    print(f"promoted: {result.promoted} — {result.reason}")
    if result.promoted_ref:
        print(f"prompt registry: {result.promoted_ref} (tag: {args.tag})")
    return 0


def cmd_optimize_reflective(args: argparse.Namespace) -> int:
    """GEPA-style reflective prompt optimization against a dataset."""
    from ..evals.datasets import Dataset
    from ..optimize.search import FitnessWeights

    app = _load_app(args.app)
    dataset = Dataset.load(args.dataset)
    weights = FitnessWeights()
    if args.target == "groundedness":
        weights.groundedness = 2.0
    elif args.target == "cost":
        weights.cost = 2.0
    elif args.target == "quality":
        weights.accuracy = 2.0
    metrics = ["semantic_similarity", "groundedness", "schema_validity", "cost", "latency"]
    result = app.reflective_optimize(
        dataset,
        strategy=args.strategy,
        metrics=metrics,
        budget=args.budget,
        minibatch_size=args.minibatch,
        seed=args.seed,
        weights=weights,
        concurrency=args.concurrency,
        apply=args.apply,
    )
    print(f"strategy: {result.strategy} · baseline fitness: {result.baseline_fitness:.4f}")
    for reflection in result.reflections:
        if reflection.edits:
            print(f"  reflect: {reflection.diagnosis}")
    print(f"rollouts: {result.evaluations} · frontier: "
          f"{len(result.frontier.front) if result.frontier else 0} non-dominated")
    print(f"promoted: {result.promoted} — {result.reason}")
    if result.promoted and result.best is not None and args.output:
        Path(args.output).write_text(
            yaml.safe_dump(result.best.payload.spec.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )
        print(f"wrote winning prompt spec to {args.output}")
    return 0


def cmd_distill(args: argparse.Namespace) -> int:
    """Curate captured traces into grounded fine-tuning JSONL."""
    from ..observability.exporters import JSONLExporter
    from ..optimize.distill import export_training_set

    exporter = JSONLExporter(args.traces_dir)
    traces = exporter.load_all()
    training_set = export_training_set(
        traces,
        name=Path(args.output).stem,
        min_feedback_score=args.min_feedback,
        require_grounding=not args.allow_ungrounded,
        min_support=args.min_support,
        max_examples=args.max_examples,
    )
    training_set.save(args.output, format=args.format)
    dropped = training_set.metadata.get("dropped_ungrounded", 0)
    print(
        f"wrote {len(training_set)} example(s) ({args.format} format) from {len(traces)} trace(s) "
        f"to {args.output}; dropped {dropped} ungrounded; grounded "
        f"fraction {training_set.grounded_fraction}"
    )
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


def cmd_audit_verify(args: argparse.Namespace) -> int:
    from ..security.audit import verify_audit_file

    path = Path(args.path)
    if not path.is_file():
        return _fail(f"audit log not found: {path}")
    result = verify_audit_file(path)
    if args.json:
        print(json_dumps(result.model_dump()))
        return 0 if result.intact else 1
    if result.intact:
        print(f"OK  hash chain intact over {result.entries} entries ({path})")
        return 0
    print(
        f"TAMPERED  chain broke at line {result.broken_at}: {result.reason} "
        f"(verified {result.entries} entries before the break)",
        file=sys.stderr,
    )
    return 1


def _governance_output(payload: Any, output: str | None) -> int:
    text = payload if isinstance(payload, str) else json_dumps(payload, indent=2)
    if output:
        Path(output).write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
        print(f"saved to {output}")
    else:
        print(text)
    return 0


def cmd_governance_card(args: argparse.Namespace) -> int:
    app = _load_app(args.app)
    fmt = args.format
    if args.kind == "system":
        card = app.system_card(format=fmt)
    else:
        card = app.model_card(format=fmt)
    return _governance_output(card.to_json(), args.output)


def cmd_governance_report(args: argparse.Namespace) -> int:
    app = _load_app(args.app)
    redteam = None
    if args.red_team:
        from ..evals.redteam import RedTeamSuite

        redteam = RedTeamSuite().run(app)
    report = app.compliance_report(redteam=redteam)
    if args.markdown:
        return _governance_output(report.to_markdown(), args.output)
    payload = report.model_dump(mode="json") if args.full else report.summary()
    return _governance_output(payload, args.output)


def cmd_governance_aibom(args: argparse.Namespace) -> int:
    app = _load_app(args.app)
    bom = app.aibom()
    return _governance_output(bom.to_json(), args.output)


def cmd_governance_lineage(args: argparse.Namespace) -> int:
    app = _load_app(args.app)
    record = app.trace_lineage(args.source)
    if record.is_empty:
        return _fail(f"no lineage for source {args.source!r} (ingest a source first)")
    return _governance_output(record.model_dump(mode="json"), args.output)


def cmd_governance_erase(args: argparse.Namespace) -> int:
    app = _load_app(args.app)
    result = app.erase_source(args.source)
    print(
        f"erase {args.source!r}: found={result.found} "
        f"chunks={result.chunks_removed} documents={result.documents_removed} "
        f"memories={result.memories_removed} indexes_swept={result.indexes_swept}"
    )
    if result.audit_entry_id:
        print(f"audit: {result.audit_entry_id}")
    return 0 if result.found else 1


def _mcp_client(args: argparse.Namespace):
    import shlex

    from ..mcp import connect_http, connect_stdio

    if getattr(args, "command", None):
        return connect_stdio(shlex.split(args.command))
    if getattr(args, "url", None):
        return connect_http(args.url)
    raise VincioError("provide --command (stdio) or --url (Streamable HTTP)")


def cmd_mcp_tools(args: argparse.Namespace) -> int:
    from ..providers.base import run_sync

    client = _mcp_client(args)

    async def go():
        tools = await client.list_tools()
        resources = await client.list_resources() if args.resources else []
        await client.aclose()
        return tools, resources

    tools, resources = run_sync(go())
    if args.json:
        print(
            json_dumps(
                {
                    "tools": [t.model_dump() for t in tools],
                    "resources": [r.model_dump() for r in resources],
                }
            )
        )
        return 0
    print(f"{len(tools)} tool(s):")
    for tool in tools:
        print(f"  {tool.name} — {tool.description}")
    if args.resources:
        print(f"{len(resources)} resource(s):")
        for resource in resources:
            print(f"  {resource.uri} ({resource.mime_type})")
    return 0


def cmd_mcp_add(args: argparse.Namespace) -> int:
    import shlex

    app = _load_app(args.app)
    before = set(app.enabled_tools)
    app.add_mcp_server(
        args.name,
        command=shlex.split(args.command) if args.command else None,
        url=args.url,
        resources=args.resources,
    )
    added = [t for t in app.enabled_tools if t not in before]
    print(f"connected MCP server {args.name!r}; registered {len(added)} tool(s): {', '.join(added)}")
    return 0


def cmd_mcp_serve(args: argparse.Namespace) -> int:
    from ..mcp import serve_stdio
    from ..providers.base import run_sync

    app = _load_app(args.app)
    server = app.serve_mcp(name=args.name)
    print(f"serving app {server.name!r} over MCP (stdio); reading JSON-RPC on stdin…", file=sys.stderr)
    run_sync(serve_stdio(server))
    return 0


# -- parser ----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vincio", description="Vincio context engineering platform")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="initialize a vincio project")
    p_init.add_argument("path", nargs="?", default=".")
    p_init.add_argument("--project", default=None)
    p_init.add_argument(
        "--template", default="minimal", choices=["minimal", "rag", "agent", "eval"],
        help="project scaffold to generate",
    )
    p_init.add_argument("--provider", default="openai", help="default provider for vincio.yaml")
    p_init.add_argument("--force", action="store_true")
    p_init.set_defaults(fn=cmd_init)

    p_config = sub.add_parser("config", help="configuration tooling")
    config_sub = p_config.add_subparsers(dest="config_command", required=True)
    p_cfg_schema = config_sub.add_parser("schema", help="emit the vincio.yaml JSON Schema")
    p_cfg_schema.add_argument("--output", default=None, help="write schema JSON here")
    p_cfg_schema.set_defaults(fn=cmd_config_schema)
    p_cfg_validate = config_sub.add_parser("validate", help="validate a vincio config file")
    p_cfg_validate.add_argument("path", nargs="?", default=None)
    p_cfg_validate.set_defaults(fn=cmd_config_validate)
    p_cfg_show = config_sub.add_parser("show", help="print the effective merged config")
    p_cfg_show.add_argument("path", nargs="?", default=None)
    p_cfg_show.set_defaults(fn=cmd_config_show)

    p_packs = sub.add_parser("packs", help="domain packs")
    packs_sub = p_packs.add_subparsers(dest="packs_command", required=True)
    p_packs_list = packs_sub.add_parser("list", help="list available domain packs")
    p_packs_list.set_defaults(fn=cmd_packs_list)
    p_packs_show = packs_sub.add_parser("show", help="show a pack's configuration")
    p_packs_show.add_argument("name")
    p_packs_show.set_defaults(fn=cmd_packs_show)

    p_tui = sub.add_parser("tui", help="interactive inspector for runs, traces, and memory")
    p_tui.add_argument("--traces-dir", default=".vincio/traces")
    p_tui.add_argument("--db", default=".vincio/memory.db", help="memory database")
    p_tui.set_defaults(fn=cmd_tui)

    p_run = sub.add_parser("run", help="run an app once")
    p_run.add_argument("app", help="python file exposing a ContextApp as `app`")
    p_run.add_argument("--input", required=True)
    p_run.add_argument("--file", action="append", help="attach a file (repeatable)")
    p_run.add_argument("--tenant", default=None)
    p_run.add_argument("--user", default=None)
    p_run.set_defaults(fn=cmd_run)

    p_batch = sub.add_parser("batch", help="run a set of inputs through a Batch API (~50% cost)")
    p_batch.add_argument("app", help="python file exposing a ContextApp as `app`")
    p_batch.add_argument("--input", action="append", help="an input (repeatable)")
    p_batch.add_argument("--input-file", default=None, help="file with one input per line")
    p_batch.add_argument("--discount", type=float, default=0.5, help="batch price multiplier")
    p_batch.add_argument("--output", default=None, help="save results JSON here")
    p_batch.set_defaults(fn=cmd_batch)

    p_cost = sub.add_parser("cost", help="cost attribution reporting")
    cost_sub = p_cost.add_subparsers(dest="cost_command", required=True)
    p_cost_report = cost_sub.add_parser("report", help="roll up attributed cost by a dimension")
    p_cost_report.add_argument(
        "--by", default="tenant",
        choices=["tenant", "feature", "user", "model", "provider", "run"],
    )
    p_cost_report.add_argument("--db", default=".vincio/vincio.db", help="metadata database")
    p_cost_report.add_argument("--json", action="store_true", help="emit JSON")
    p_cost_report.set_defaults(fn=cmd_cost_report)

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
    p_eval_dataset = eval_sub.add_parser("dataset", help="build a dataset from captured traces")
    p_eval_dataset.add_argument("output", help="output JSONL path")
    p_eval_dataset.add_argument("--traces-dir", default=".vincio/traces")
    p_eval_dataset.add_argument("--name", default=None, help="dataset name")
    p_eval_dataset.add_argument(
        "--min-feedback", type=float, default=None,
        help="keep only traces with mean feedback score >= this",
    )
    p_eval_dataset.add_argument(
        "--group-by-session", action="store_true",
        help="stitch traces of one session into a multi-turn case",
    )
    p_eval_dataset.set_defaults(fn=cmd_eval_dataset)
    p_eval_drift = eval_sub.add_parser("drift", help="detect metric drift between two reports")
    p_eval_drift.add_argument("baseline", help="baseline report JSON")
    p_eval_drift.add_argument("current", help="current report JSON")
    p_eval_drift.add_argument("--metric", action="append", help="restrict to metric (repeatable)")
    p_eval_drift.add_argument("--threshold", type=float, default=0.1, help="mean-shift drift threshold")
    p_eval_drift.add_argument("--output", default=None, help="save drift records JSON here")
    p_eval_drift.set_defaults(fn=cmd_eval_drift)
    p_eval_annotate = eval_sub.add_parser("annotate", help="track human↔judge agreement (Cohen's κ)")
    p_eval_annotate.add_argument("labels", help="JSONL of {judge, human} score pairs (or a dataset)")
    p_eval_annotate.add_argument("--threshold", type=float, default=0.6, help="κ needed for CI gating weight")
    p_eval_annotate.add_argument("--bins", type=int, default=2, help="number of score bins for κ")
    p_eval_annotate.set_defaults(fn=cmd_eval_annotate)
    p_eval_regress = eval_sub.add_parser(
        "regress", help="swap only the model and test for a statistically grounded regression"
    )
    p_eval_regress.add_argument("dataset")
    p_eval_regress.add_argument("--app", required=True)
    p_eval_regress.add_argument("--baseline-model", default=None, dest="baseline_model",
                                help="defaults to the app's configured model")
    p_eval_regress.add_argument("--candidate-model", required=True, dest="candidate_model")
    p_eval_regress.add_argument("--metric", action="append", help="metric name (repeatable)")
    p_eval_regress.add_argument("--quality-metric", default="semantic_similarity",
                                dest="quality_metric")
    p_eval_regress.add_argument("--alpha", type=float, default=0.05)
    p_eval_regress.add_argument("--repeats", type=int, default=1,
                                help="runs per case for mean/stdev + flake quarantine")
    p_eval_regress.add_argument("--no-flake-quarantine", action="store_true",
                                dest="no_flake_quarantine")
    p_eval_regress.add_argument("--output", default=None, help="save the regression report JSON")
    p_eval_regress.set_defaults(fn=cmd_eval_regress)

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
    p_push = prompt_sub.add_parser("push", help="version a prompt spec in the registry")
    p_push.add_argument("path", help="prompt spec file (yaml/json)")
    p_push.add_argument("--name", default=None, help="registry name (defaults to spec name)")
    p_push.add_argument("--tag", action="append", help="tag to apply (repeatable)")
    p_push.add_argument("--message", default=None, help="version message")
    p_push.add_argument("--registry", default=".vincio/prompts")
    p_push.set_defaults(fn=cmd_prompt_push)
    p_versions = prompt_sub.add_parser("versions", help="list versions of a prompt")
    p_versions.add_argument("name")
    p_versions.add_argument("--registry", default=".vincio/prompts")
    p_versions.set_defaults(fn=cmd_prompt_versions)
    p_pdiff = prompt_sub.add_parser("diff", help="diff two prompt versions")
    p_pdiff.add_argument("name")
    p_pdiff.add_argument("version_a", type=int)
    p_pdiff.add_argument("version_b", type=int)
    p_pdiff.add_argument("--rendered", action="store_true", help="include rendered text diff")
    p_pdiff.add_argument("--registry", default=".vincio/prompts")
    p_pdiff.set_defaults(fn=cmd_prompt_diff)
    p_rollback = prompt_sub.add_parser("rollback", help="re-publish an earlier version as head")
    p_rollback.add_argument("name")
    p_rollback.add_argument("--to", type=int, default=None, help="version to roll back to")
    p_rollback.add_argument("--registry", default=".vincio/prompts")
    p_rollback.set_defaults(fn=cmd_prompt_rollback)

    p_trace = sub.add_parser("trace", help="trace tooling")
    trace_sub = p_trace.add_subparsers(dest="trace_command", required=True)
    for name, fn, extra in (
        ("show", cmd_trace_show, ["trace_id"]),
        ("view", cmd_trace_view, ["trace_id"]),
        ("replay", cmd_trace_replay, ["trace_id"]),
        ("diff", cmd_trace_diff, ["trace_a", "trace_b"]),
    ):
        p_trace_sub = trace_sub.add_parser(name)
        for argument in extra:
            p_trace_sub.add_argument(argument)
        p_trace_sub.add_argument("--traces-dir", default=".vincio/traces")
        if name == "diff":
            p_trace_sub.add_argument("--html", default=None, help="write a visual diff HTML here")
        if name == "replay":
            p_trace_sub.add_argument(
                "--against", default=None,
                help="path to an app file to replay this trace against (diffs output/trajectory/cost)",
            )
            p_trace_sub.add_argument(
                "--pin-tools", action="store_true", dest="pin_tools",
                help="pin recorded tool outputs for a deterministic replay",
            )
        p_trace_sub.set_defaults(fn=fn)
    p_trace_export = trace_sub.add_parser("export", help="export a trace/session as static HTML")
    p_trace_export.add_argument("trace_id", help="trace id (or session id with --session)")
    p_trace_export.add_argument("--session", action="store_true", help="export a whole session")
    p_trace_export.add_argument("--output", default=None, help="output HTML path")
    p_trace_export.add_argument("--traces-dir", default=".vincio/traces")
    p_trace_export.set_defaults(fn=cmd_trace_export)
    p_trace_sessions = trace_sub.add_parser("sessions", help="list sessions with aggregates")
    p_trace_sessions.add_argument("--traces-dir", default=".vincio/traces")
    p_trace_sessions.set_defaults(fn=cmd_trace_sessions)
    p_trace_feedback = trace_sub.add_parser("feedback", help="attach feedback to a trace")
    p_trace_feedback.add_argument("trace_id")
    p_trace_feedback.add_argument("--key", default="user_rating")
    p_trace_feedback.add_argument("--score", type=float, default=None)
    p_trace_feedback.add_argument("--comment", default=None)
    p_trace_feedback.add_argument("--user", default=None)
    p_trace_feedback.add_argument("--traces-dir", default=".vincio/traces")
    p_trace_feedback.set_defaults(fn=cmd_trace_feedback)

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

    p_opt_refl = optimize_sub.add_parser(
        "reflective", help="GEPA-style reflective prompt optimization (1.4)"
    )
    p_opt_refl.add_argument("--app", required=True)
    p_opt_refl.add_argument("--dataset", required=True)
    p_opt_refl.add_argument("--strategy", default="reflective", choices=["reflective", "mipro"])
    p_opt_refl.add_argument("--target", default="quality", choices=["quality", "groundedness", "cost"])
    p_opt_refl.add_argument("--budget", type=int, default=12, help="max evaluation rollouts")
    p_opt_refl.add_argument("--minibatch", type=int, default=8, help="screening minibatch size")
    p_opt_refl.add_argument("--seed", type=int, default=7)
    p_opt_refl.add_argument("--concurrency", type=int, default=4)
    p_opt_refl.add_argument("--apply", action="store_true", help="apply the winner to the app spec")
    p_opt_refl.add_argument("--output", default=None, help="write winning spec YAML here")
    p_opt_refl.set_defaults(fn=cmd_optimize_reflective)

    p_distill = sub.add_parser("distill", help="curate traces into grounded fine-tuning JSONL (1.4)")
    p_distill.add_argument("--traces-dir", default=".vincio/traces", help="directory of captured traces")
    p_distill.add_argument("--output", required=True, help="output JSONL path")
    p_distill.add_argument("--format", default="openai", choices=["openai", "anthropic"])
    p_distill.add_argument("--min-feedback", type=float, default=None, help="min mean feedback score")
    p_distill.add_argument("--min-support", type=float, default=0.5, help="evidence support a claim needs")
    p_distill.add_argument("--max-examples", type=int, default=None)
    p_distill.add_argument(
        "--allow-ungrounded", action="store_true",
        help="keep examples without evidence support (default: drop them)",
    )
    p_distill.set_defaults(fn=cmd_distill)

    p_loop = sub.add_parser("loop", help="closed improvement loop (0.8)")
    loop_sub = p_loop.add_subparsers(dest="loop_command", required=True)
    p_loop_run = loop_sub.add_parser(
        "run", help="trace → dataset → eval → optimize → promote, one cycle"
    )
    p_loop_run.add_argument("--app", required=True)
    p_loop_run.add_argument("--dataset", default=None, help="JSONL dataset (default: curate from captured traces)")
    p_loop_run.add_argument("--min-feedback", type=float, default=None, help="min mean feedback score when curating from traces")
    p_loop_run.add_argument("--budget", type=int, default=8, help="max prompt variants")
    p_loop_run.add_argument("--subset", type=int, default=8, help="screening subset size")
    p_loop_run.add_argument("--concurrency", type=int, default=4)
    p_loop_run.add_argument("--gate", action="append", help="promotion gate, e.g. groundedness='>= 0.8'")
    p_loop_run.add_argument("--tag", default="production", help="registry tag for the promoted version")
    p_loop_run.add_argument("--experiment", default="improvement_loop")
    p_loop_run.add_argument("--dry-run", action="store_true", help="report the decision without promoting")
    p_loop_run.add_argument(
        "--reflective", action="store_true", help="use the GEPA-style reflective optimizer (1.4)"
    )
    p_loop_run.set_defaults(fn=cmd_loop_run)

    p_index = sub.add_parser("index", help="index commands")
    index_sub = p_index.add_subparsers(dest="index_command", required=True)
    p_index_build = index_sub.add_parser("build", help="load, chunk, and persist documents")
    p_index_build.add_argument("path")
    p_index_build.add_argument("--db", default=".vincio/vincio.db")
    p_index_build.add_argument("--chunking", default="adaptive")
    p_index_build.add_argument("--chunk-size", type=int, default=400)
    p_index_build.set_defaults(fn=cmd_index_build)

    p_audit = sub.add_parser("audit", help="audit-log integrity tooling")
    audit_sub = p_audit.add_subparsers(dest="audit_command", required=True)
    p_audit_verify = audit_sub.add_parser(
        "verify", help="verify the hash chain of a persisted audit JSONL file"
    )
    p_audit_verify.add_argument("path", nargs="?", default=".vincio/audit/audit.jsonl")
    p_audit_verify.add_argument("--json", action="store_true", help="emit the result as JSON")
    p_audit_verify.set_defaults(fn=cmd_audit_verify)

    p_gov = sub.add_parser("governance", help="enterprise governance & compliance artifacts")
    gov_sub = p_gov.add_subparsers(dest="governance_command", required=True)
    p_gov_card = gov_sub.add_parser("card", help="generate a model or system card")
    p_gov_card.add_argument("app", help="python file exposing a ContextApp as `app`")
    p_gov_card.add_argument("--kind", choices=["model", "system"], default="system")
    p_gov_card.add_argument(
        "--format", choices=["vincio", "open_model_card", "ai_card"], default="vincio"
    )
    p_gov_card.add_argument("--output", default=None, help="write the card JSON here")
    p_gov_card.set_defaults(fn=cmd_governance_card)
    p_gov_report = gov_sub.add_parser(
        "report", help="emit the OWASP/NIST/MITRE compliance coverage matrix"
    )
    p_gov_report.add_argument("app", help="python file exposing a ContextApp as `app`")
    p_gov_report.add_argument("--red-team", action="store_true", help="run the red-team suite for evidence")
    p_gov_report.add_argument("--full", action="store_true", help="emit the full per-control matrix")
    p_gov_report.add_argument("--markdown", action="store_true", help="emit a Markdown matrix")
    p_gov_report.add_argument("--output", default=None, help="write the report here")
    p_gov_report.set_defaults(fn=cmd_governance_report)
    p_gov_aibom = gov_sub.add_parser("aibom", help="generate an AI bill of materials (CycloneDX)")
    p_gov_aibom.add_argument("app", help="python file exposing a ContextApp as `app`")
    p_gov_aibom.add_argument("--output", default=None, help="write the AI-BOM JSON here")
    p_gov_aibom.set_defaults(fn=cmd_governance_aibom)
    p_gov_lineage = gov_sub.add_parser("lineage", help="trace a source's lineage chain")
    p_gov_lineage.add_argument("app", help="python file exposing a ContextApp as `app`")
    p_gov_lineage.add_argument("source", help="source name or document id")
    p_gov_lineage.add_argument("--output", default=None, help="write the lineage JSON here")
    p_gov_lineage.set_defaults(fn=cmd_governance_lineage)
    p_gov_erase = gov_sub.add_parser("erase", help="right-to-erasure: purge a source everywhere")
    p_gov_erase.add_argument("app", help="python file exposing a ContextApp as `app`")
    p_gov_erase.add_argument("source", help="source name or document id to erase")
    p_gov_erase.set_defaults(fn=cmd_governance_erase)

    p_mcp = sub.add_parser("mcp", help="Model Context Protocol client/server")
    mcp_sub = p_mcp.add_subparsers(dest="mcp_command", required=True)
    p_mcp_tools = mcp_sub.add_parser("tools", help="list an MCP server's tools/resources")
    p_mcp_tools.add_argument("--command", default=None, help="stdio server command, e.g. 'python server.py'")
    p_mcp_tools.add_argument("--url", default=None, help="Streamable HTTP server URL")
    p_mcp_tools.add_argument("--resources", action="store_true", help="also list resources")
    p_mcp_tools.add_argument("--json", action="store_true", help="emit JSON")
    p_mcp_tools.set_defaults(fn=cmd_mcp_tools)
    p_mcp_add = mcp_sub.add_parser("add", help="connect an MCP server to an app and register its tools")
    p_mcp_add.add_argument("app", help="python file exposing a ContextApp as `app`")
    p_mcp_add.add_argument("--name", required=True, help="name for the MCP server (tool namespace)")
    p_mcp_add.add_argument("--command", default=None, help="stdio server command")
    p_mcp_add.add_argument("--url", default=None, help="Streamable HTTP server URL")
    p_mcp_add.add_argument("--resources", action="store_true", help="also import resources as evidence")
    p_mcp_add.set_defaults(fn=cmd_mcp_add)
    p_mcp_serve = mcp_sub.add_parser("serve", help="expose an app as an MCP server over stdio")
    p_mcp_serve.add_argument("app", help="python file exposing a ContextApp as `app`")
    p_mcp_serve.add_argument("--name", default=None, help="server name (defaults to the app name)")
    p_mcp_serve.set_defaults(fn=cmd_mcp_serve)

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

    p_providers = sub.add_parser("providers", help="provider/model rotation, lifecycle & swap gate")
    providers_sub = p_providers.add_subparsers(dest="providers_command", required=True)

    p_prov_list = providers_sub.add_parser("list", help="list the model registry catalog")
    p_prov_list.add_argument("--provider", default=None, help="filter by provider")
    p_prov_list.add_argument("--json", action="store_true")
    p_prov_list.set_defaults(fn=cmd_providers_list)

    p_prov_lifecycle = providers_sub.add_parser(
        "lifecycle", help="scan pinned models for sunset and propose migrations"
    )
    p_prov_lifecycle.add_argument("--app", default=None, help="app file (uses its pinned models)")
    p_prov_lifecycle.add_argument("--model", action="append", help="model id (repeatable)")
    p_prov_lifecycle.add_argument("--as-of", default=None, dest="as_of", help="YYYY-MM-DD")
    p_prov_lifecycle.add_argument("--warn-within-days", type=int, default=90,
                                  dest="warn_within_days")
    p_prov_lifecycle.add_argument("--json", action="store_true")
    p_prov_lifecycle.set_defaults(fn=cmd_providers_lifecycle)

    p_prov_discover = providers_sub.add_parser(
        "discover", help="reconcile a provider's live model list into the registry"
    )
    p_prov_discover.add_argument("provider", help="provider name (openai/anthropic/google/...)")
    p_prov_discover.add_argument("--mark-missing-deprecated", action="store_true",
                                 dest="mark_missing_deprecated")
    p_prov_discover.add_argument("--json", action="store_true")
    p_prov_discover.set_defaults(fn=cmd_providers_discover)

    p_prov_regress = providers_sub.add_parser(
        "regress", help="gate a model swap (replay + eval + cost/latency/behavioral diff)"
    )
    p_prov_regress.add_argument("--app", required=True)
    p_prov_regress.add_argument("--candidate-model", required=True, dest="candidate_model")
    p_prov_regress.add_argument("--baseline-model", default=None, dest="baseline_model")
    p_prov_regress.add_argument("--dataset", default=None, help="golden dataset JSONL")
    p_prov_regress.add_argument("--trace", action="append", help="golden trace id (repeatable)")
    p_prov_regress.add_argument("--traces-dir", default=".vincio/traces")
    p_prov_regress.add_argument("--gate", action="append", help="gate, e.g. groundedness='>= 0.9'")
    p_prov_regress.add_argument("--quality-metric", default="semantic_similarity",
                                dest="quality_metric")
    p_prov_regress.add_argument("--alpha", type=float, default=0.05)
    p_prov_regress.add_argument("--repeats", type=int, default=1)
    p_prov_regress.set_defaults(fn=cmd_providers_regress)

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
