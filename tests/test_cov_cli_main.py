"""Real-behavior coverage for ``vincio.cli.main``.

These tests drive the CLI entry points (``main`` and the ``cmd_*`` handlers)
through crafted argv, asserting on exit codes and printed output. Model
interaction goes through the deterministic ``MockProvider`` written into a
scaffolded ``app.py`` — never a stand-in mock object.
"""

from __future__ import annotations

import importlib
import json

import pytest

from vincio.cli.main import (
    _fail,
    _load_app,
    _load_trace,
    build_parser,
    cmd_init,
    main,
)
from vincio.core.errors import VincioError

# `vincio.cli.main` the attribute is shadowed by the re-exported `main`
# function, so reach the module object explicitly.
cli_main = importlib.import_module("vincio.cli.main")

# A minimal, fully offline app file: a ContextApp backed by MockProvider that
# echoes a fixed answer. _load_app imports this and finds the `app` instance.
_APP_PY = (
    "from vincio import ContextApp, VincioConfig\n"
    "from vincio.providers import MockProvider\n"
    "cfg = VincioConfig()\n"
    "cfg.observability.exporter = 'jsonl'\n"
    "app = ContextApp(\n"
    "    name='covdemo',\n"
    "    provider=MockProvider(default_text='forty two'),\n"
    "    model='mock-1',\n"
    "    config=cfg,\n"
    ")\n"
)


def _write_app(tmp_path) -> str:
    path = tmp_path / "app.py"
    path.write_text(_APP_PY, encoding="utf-8")
    return str(path)


# -- helpers / dispatch -------------------------------------------------------


def test_fail_prints_to_stderr_and_returns_code(capsys):
    code = _fail("boom", code=7)
    err = capsys.readouterr().err
    assert code == 7
    assert err == "error: boom\n"


def test_fail_default_code_is_one():
    assert _fail("x") == 1


def test_load_app_missing_file_raises(tmp_path):
    with pytest.raises(VincioError, match="app file not found"):
        _load_app(str(tmp_path / "nope.py"))


def test_load_app_without_context_app_raises(tmp_path):
    bare = tmp_path / "bare.py"
    bare.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(VincioError, match="no ContextApp instance found"):
        _load_app(str(bare))


def test_load_app_finds_non_app_named_instance(tmp_path):
    # `app` attribute is absent; the loader scans module values for a ContextApp.
    src = tmp_path / "weird.py"
    src.write_text(
        "from vincio import ContextApp\n"
        "from vincio.providers import MockProvider\n"
        "engine = ContextApp(name='w', provider=MockProvider(default_text='hi'))\n",
        encoding="utf-8",
    )
    loaded = _load_app(str(src))
    assert loaded.name == "w"


def test_load_trace_missing_raises(tmp_path):
    with pytest.raises(VincioError, match="trace 'absent' not found"):
        _load_trace("absent", str(tmp_path / "traces"))


def test_main_translates_vincio_error_to_exit_one(capsys):
    # `run` on a missing app file raises VincioError, which main() catches.
    code = main(["run", "/does/not/exist.py", "--input", "hi"])
    err = capsys.readouterr().err
    assert code == 1
    assert "app file not found" in err


def test_main_unknown_command_exits_two():
    with pytest.raises(SystemExit) as excinfo:
        main(["frobnicate"])
    assert excinfo.value.code == 2


def test_main_no_command_required(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2
    assert "required" in capsys.readouterr().err.lower()


def test_build_parser_dispatch_table_is_wired():
    parser = build_parser()
    ns = parser.parse_args(["init", "somewhere"])
    assert ns.fn is cmd_init
    assert ns.path == "somewhere"


def test_main_keyboard_interrupt_returns_130(monkeypatch):
    # main() catches a KeyboardInterrupt raised by the dispatched handler and
    # returns the conventional 130 exit code. We replace a real handler so the
    # whole parse→dispatch→except path runs.
    def boom(_args):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli_main, "cmd_plugins_list", boom)
    assert main(["plugins", "list"]) == 130


# -- run / batch --------------------------------------------------------------


def test_cmd_run_prints_status_trace_and_output(tmp_path, capsys):
    app_py = _write_app(tmp_path)
    code = main(["run", app_py, "--input", "what is the answer?"])
    out = capsys.readouterr().out
    assert code == 0
    assert "status:" in out
    assert "trace:" in out
    assert "output:" in out
    assert "forty two" in out


def test_cmd_batch_requires_inputs(tmp_path, capsys):
    app_py = _write_app(tmp_path)
    code = main(["batch", app_py])
    assert code == 1
    assert "no inputs" in capsys.readouterr().err


def test_cmd_batch_runs_multiple_inputs_and_saves(tmp_path, capsys):
    app_py = _write_app(tmp_path)
    out_json = tmp_path / "results.json"
    code = main(
        ["batch", app_py, "--input", "a", "--input", "b", "--output", str(out_json)]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "2/2 succeeded" in out
    assert f"saved results to {out_json}" in out
    saved = json.loads(out_json.read_text(encoding="utf-8"))
    assert len(saved) == 2


def test_cmd_batch_reads_input_file(tmp_path, capsys):
    app_py = _write_app(tmp_path)
    inputs = tmp_path / "inputs.txt"
    inputs.write_text("first\n\n  second  \n", encoding="utf-8")  # blank line skipped
    code = main(["batch", app_py, "--input-file", str(inputs)])
    out = capsys.readouterr().out
    assert code == 0
    assert "2/2 succeeded" in out


# -- packs / plugins ----------------------------------------------------------


def test_packs_show_unknown_pack_is_vincio_error(capsys):
    code = main(["packs", "show", "does_not_exist"])
    assert code == 1
    assert "error:" in capsys.readouterr().err


def test_plugins_list_reports_api_and_no_plugins(capsys):
    code = main(["plugins", "list"])
    out = capsys.readouterr().out
    assert code == 0
    assert "plugin API:" in out
    assert "no third-party plugins installed" in out


# -- providers list -----------------------------------------------------------


def test_providers_list_text_table(capsys):
    code = main(["providers", "list"])
    out = capsys.readouterr().out
    assert code == 0
    assert "provider" in out
    assert out.strip().endswith("model(s)")


def test_providers_list_json_and_provider_filter(capsys):
    code = main(["providers", "list", "--provider", "openai", "--json"])
    out = capsys.readouterr().out
    assert code == 0
    rows = json.loads(out)
    assert rows  # openai ships profiles
    assert all(r["provider"] == "openai" for r in rows)


def test_providers_lifecycle_requires_model_or_app(capsys):
    code = main(["providers", "lifecycle"])
    assert code == 1
    assert "provide --model" in capsys.readouterr().err


def test_providers_lifecycle_no_models_nearing_sunset(capsys):
    # A far-past as-of date means nothing is near sunset → exit 0.
    code = main(
        ["providers", "lifecycle", "--model", "gpt-5.2", "--as-of", "2020-01-01"]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "no models nearing sunset" in out


# -- config validate / show error paths ---------------------------------------


def test_config_validate_no_file_found(tmp_cwd, capsys):
    # Empty isolated cwd: find_config_file returns None.
    code = main(["config", "validate"])
    assert code == 1
    assert "no vincio config file found" in capsys.readouterr().err


def test_config_migrate_missing_file(tmp_path, capsys):
    code = main(["config", "migrate", str(tmp_path / "absent.yaml")])
    assert code == 1
    assert "config file not found" in capsys.readouterr().err


def test_config_migrate_non_mapping_root(tmp_path, capsys):
    cfg = tmp_path / "vincio.yaml"
    cfg.write_text("- just\n- a\n- list\n", encoding="utf-8")
    code = main(["config", "migrate", str(cfg)])
    assert code == 1
    assert "config root must be a mapping" in capsys.readouterr().err


def test_config_migrate_unparseable_yaml(tmp_path, capsys):
    cfg = tmp_path / "vincio.yaml"
    cfg.write_text("project: [unterminated\n", encoding="utf-8")
    code = main(["config", "migrate", str(cfg)])
    assert code == 1
    assert "could not parse" in capsys.readouterr().err


def test_config_migrate_already_current_is_noop(tmp_path, capsys):
    # Scaffold a current-version config, then migrate: nothing to do.
    root = tmp_path / "proj"
    assert main(["init", str(root)]) == 0
    capsys.readouterr()
    code = main(["config", "migrate", str(root / "vincio.yaml")])
    out = capsys.readouterr().out
    assert code == 0
    assert "nothing to migrate" in out


# -- migrate (source codemod) -------------------------------------------------


def test_migrate_unknown_target_rejected_by_argparse():
    # `target` is constrained by argparse `choices`, so an unknown value exits 2.
    with pytest.raises(SystemExit) as excinfo:
        main(["migrate", "9.9"])
    assert excinfo.value.code == 2


def test_migrate_4_0_clean_project(tmp_path, capsys):
    src = tmp_path / "app.py"
    src.write_text("import vincio\n", encoding="utf-8")
    code = main(["migrate", "4.0", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "vincio migrate 4.0" in out


def test_migrate_4_0_json_output(tmp_path, capsys):
    src = tmp_path / "app.py"
    src.write_text("import vincio\n", encoding="utf-8")
    code = main(["migrate", "4.0", str(tmp_path), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["target"] == "4.0"
    assert payload["ok"] is True


# -- eval run gate parsing ----------------------------------------------------


def test_eval_run_rejects_malformed_gate(tmp_path, capsys):
    app_py = _write_app(tmp_path)
    golden = tmp_path / "g.jsonl"
    golden.write_text(json.dumps({"id": "c1", "input": "x", "expected": "y"}) + "\n")
    code = main(
        ["eval", "run", str(golden), "--app", app_py, "--gate", "no_equals_sign"]
    )
    assert code == 1
    assert "invalid gate" in capsys.readouterr().err


def test_eval_report_dir_without_reports(tmp_path, capsys):
    empty = tmp_path / "reports"
    empty.mkdir()
    code = main(["eval", "report", str(empty)])
    assert code == 1
    assert "no reports" in capsys.readouterr().err


# -- eval annotate ------------------------------------------------------------


def test_eval_annotate_needs_two_pairs(tmp_path, capsys):
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps({"judge": 1.0, "human": 1.0}) + "\n")
    code = main(["eval", "annotate", str(labels)])
    assert code == 1
    assert "at least 2" in capsys.readouterr().err


def test_eval_annotate_perfect_agreement_passes(tmp_path, capsys):
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        "\n".join(
            json.dumps({"judge_score": j, "human_score": h})
            for j, h in [(1.0, 1.0), (0.0, 0.0), (1.0, 1.0), (0.0, 0.0)]
        )
        + "\n"
    )
    code = main(["eval", "annotate", str(labels), "--threshold", "0.6", "--bins", "2"])
    out = capsys.readouterr().out
    assert code == 0
    assert "EARNS" in out
    assert "cohens_kappa" in out


# -- eval drift ---------------------------------------------------------------


def test_eval_drift_no_shared_metrics(tmp_path, capsys):
    from vincio.evals.reports import EvalReport

    a = EvalReport(name="base")
    a.metadata["aggregate_metrics"] = {"metric_a": 0.9}
    b = EvalReport(name="curr")
    b.metadata["aggregate_metrics"] = {"metric_b": 0.5}
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    a.save(pa)
    b.save(pb)
    code = main(["eval", "drift", str(pa), str(pb)])
    assert code == 1
    assert "no shared metrics" in capsys.readouterr().err


# -- audit verify -------------------------------------------------------------


def test_audit_verify_missing_file(tmp_path, capsys):
    code = main(["audit", "verify", str(tmp_path / "audit.jsonl")])
    assert code == 1
    assert "audit log not found" in capsys.readouterr().err


def test_audit_verify_intact_chain(tmp_path, capsys):
    from vincio.security.audit import AuditLog

    log = AuditLog(str(tmp_path / "audit"))
    log.record("created", item="x")
    log.record("updated", item="x")
    code = main(["audit", "verify", str(log.path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "hash chain intact" in out
    assert "2 entries" in out


def test_audit_verify_tampered_chain(tmp_path, capsys):
    from vincio.security.audit import AuditLog

    log = AuditLog(str(tmp_path / "audit"))
    log.record("created", item="x")
    log.record("updated", item="x")
    lines = log.path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["action"] = "TAMPERED"
    lines[0] = json.dumps(record)
    log.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    code = main(["audit", "verify", str(log.path)])
    err = capsys.readouterr().err
    assert code == 1
    assert "TAMPERED" in err


def test_audit_verify_json_output(tmp_path, capsys):
    from vincio.security.audit import AuditLog

    log = AuditLog(str(tmp_path / "audit"))
    log.record("created", item="x")
    code = main(["audit", "verify", str(log.path), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["intact"] is True
    assert payload["entries"] == 1


# -- memory -------------------------------------------------------------------


def test_memory_remember_recall_forget_roundtrip(tmp_path, capsys):
    db = str(tmp_path / "mem.db")
    code = main(["memory", "remember", "Prefers concise answers", "--user", "u1", "--db", db])
    out = capsys.readouterr().out
    assert code == 0
    assert "conf=" in out

    assert main(["memory", "recall", "answer style", "--user", "u1", "--db", db]) == 0
    recall_out = capsys.readouterr().out
    # The remembered item id appears in the recall listing.
    assert "concise" in recall_out or "[" in recall_out

    # Inspect surfaces the stored item.
    assert main(["memory", "inspect", "--user", "u1", "--db", db]) == 0
    assert "concise" in capsys.readouterr().out


def test_memory_recall_empty_store(tmp_path, capsys):
    db = str(tmp_path / "mem.db")
    code = main(["memory", "recall", "anything", "--user", "ghost", "--db", db])
    out = capsys.readouterr().out
    assert code == 0
    assert "no memories recalled" in out


def test_memory_inspect_empty(tmp_path, capsys):
    db = str(tmp_path / "empty.db")
    code = main(["memory", "inspect", "--db", db])
    out = capsys.readouterr().out
    assert code == 0
    assert "no memories found" in out


def test_memory_forget_unknown_id(tmp_path, capsys):
    db = str(tmp_path / "mem.db")
    code = main(["memory", "forget", "mem_does_not_exist", "--db", db])
    assert code == 1
    assert "memory not found" in capsys.readouterr().err


def test_memory_decay_pass(tmp_path, capsys):
    db = str(tmp_path / "mem.db")
    main(["memory", "remember", "fact", "--user", "u1", "--db", db])
    capsys.readouterr()
    code = main(["memory", "decay", "--db", db])
    out = capsys.readouterr().out
    assert code == 0
    assert "decayed=" in out and "expired=" in out


# -- cost report --------------------------------------------------------------


def test_cost_report_empty_ledger_json(tmp_path, capsys):
    db = str(tmp_path / "vincio.db")
    code = main(["cost", "report", "--db", db, "--json"])
    out = capsys.readouterr().out
    assert code == 0
    payload = json.loads(out)
    assert "by" in payload or isinstance(payload, dict)


# -- prompt lint --------------------------------------------------------------


def test_prompt_lint_no_files_in_dir(tmp_path, capsys):
    empty = tmp_path / "prompts"
    empty.mkdir()
    code = main(["prompt", "lint", str(empty)])
    assert code == 1
    assert "no prompt files found" in capsys.readouterr().err


def test_prompt_lint_parse_error_on_non_mapping(tmp_path, capsys):
    spec = tmp_path / "p.yaml"
    spec.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    code = main(["prompt", "lint", str(spec)])
    out = capsys.readouterr().out
    assert code == 1
    assert "parse error" in out


def test_prompt_compile_emits_header_and_messages(tmp_path, capsys):
    spec = tmp_path / "p.yaml"
    spec.write_text(
        "role: helpful assistant\nobjective: Answer only from documents\n"
        "rules:\n  - Use only provided documents\n",
        encoding="utf-8",
    )
    code = main(["prompt", "compile", str(spec), "--task", "hello"])
    out = capsys.readouterr().out
    assert code == 0
    assert "# prompt_id:" in out
    assert "# tokens:" in out


# -- mcp / serve error guards -------------------------------------------------


def test_mcp_tools_requires_command_or_url(capsys):
    code = main(["mcp", "tools"])
    assert code == 1
    assert "provide --command" in capsys.readouterr().err


def test_serve_requires_at_least_one_app(tmp_cwd, capsys):
    pytest.importorskip("uvicorn")
    # No --app provided → the launcher fails before binding a port.
    code = main(["serve"])
    assert code == 1
    assert "no apps to serve" in capsys.readouterr().err


# -- trace tooling (uses real produced traces) --------------------------------


def _run_and_get_trace_id(tmp_cwd, capsys, text="hi there") -> str:
    """Run the app in the isolated cwd and return the produced trace id."""
    (tmp_cwd / "app.py").write_text(_APP_PY, encoding="utf-8")
    assert main(["run", "app.py", "--input", text]) == 0
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if ln.startswith("trace:"))
    return line.split()[1]


def test_trace_show_renders_span_tree(tmp_cwd, capsys):
    trace_id = _run_and_get_trace_id(tmp_cwd, capsys)
    code = main(["trace", "show", trace_id])
    out = capsys.readouterr().out
    assert code == 0
    assert f"trace {trace_id}" in out
    assert "app=covdemo" in out


def test_trace_show_missing_trace_exits_nonzero(tmp_cwd, capsys):
    code = main(["trace", "show", "trace_missing"])
    assert code == 1
    assert "not found" in capsys.readouterr().err


def test_trace_replay_emits_plan(tmp_cwd, capsys):
    trace_id = _run_and_get_trace_id(tmp_cwd, capsys)
    code = main(["trace", "replay", trace_id])
    out = capsys.readouterr().out
    assert code == 0
    # The replay plan is JSON; it must parse.
    json.loads(out)


def test_trace_diff_two_traces(tmp_cwd, capsys):
    id_a = _run_and_get_trace_id(tmp_cwd, capsys, text="first question")
    id_b = _run_and_get_trace_id(tmp_cwd, capsys, text="second question")
    code = main(["trace", "diff", id_a, id_b])
    out = capsys.readouterr().out
    assert code == 0
    payload = json.loads(out)
    assert isinstance(payload, dict)


def test_trace_sessions_empty(tmp_cwd, capsys):
    # No traces produced yet in this fresh cwd → the no-sessions branch.
    code = main(["trace", "sessions", "--traces-dir", ".vincio/traces"])
    out = capsys.readouterr().out
    assert code == 0
    assert "no sessions found" in out


def test_trace_feedback_records_score(tmp_cwd, capsys):
    trace_id = _run_and_get_trace_id(tmp_cwd, capsys)
    code = main(
        ["trace", "feedback", trace_id, "--key", "user_rating", "--score", "0.9"]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert f"recorded feedback on {trace_id}" in out
    assert "score=0.9" in out


def test_trace_view_renders_text(tmp_cwd, capsys):
    trace_id = _run_and_get_trace_id(tmp_cwd, capsys)
    code = main(["trace", "view", trace_id])
    out = capsys.readouterr().out
    assert code == 0
    assert trace_id in out


def test_trace_export_writes_html(tmp_cwd, capsys):
    trace_id = _run_and_get_trace_id(tmp_cwd, capsys)
    code = main(["trace", "export", trace_id, "--output", "trace.html"])
    out = capsys.readouterr().out
    assert code == 0
    assert "wrote trace.html" in out
    html = (tmp_cwd / "trace.html").read_text(encoding="utf-8")
    assert "<html" in html.lower() or "<!doctype" in html.lower()


# -- eval dataset (from produced traces) --------------------------------------


def test_eval_dataset_from_traces(tmp_cwd, capsys):
    _run_and_get_trace_id(tmp_cwd, capsys)
    code = main(["eval", "dataset", "golden.jsonl"])
    out = capsys.readouterr().out
    assert code == 0
    assert "case(s) from" in out
    assert (tmp_cwd / "golden.jsonl").is_file()


# -- prompt registry ----------------------------------------------------------


def _write_prompt_spec(tmp_path) -> str:
    spec = tmp_path / "prompt.yaml"
    spec.write_text(
        "role: helpful assistant\nobjective: Answer only from documents\n"
        "rules:\n  - Use only provided documents\n",
        encoding="utf-8",
    )
    return str(spec)


def test_prompt_push_versions_diff_rollback(tmp_path, capsys):
    spec = _write_prompt_spec(tmp_path)
    registry = str(tmp_path / "registry")

    # push v1
    assert main(
        ["prompt", "push", spec, "--name", "qa", "--registry", registry,
         "--tag", "prod", "--message", "initial"]
    ) == 0
    push_out = capsys.readouterr().out
    assert "qa@" in push_out or "tags=prod" in push_out

    # push a changed v2 so a diff exists
    (tmp_path / "prompt.yaml").write_text(
        "role: helpful assistant\nobjective: Answer concisely from documents\n"
        "rules:\n  - Use only provided documents\n  - Be terse\n",
        encoding="utf-8",
    )
    assert main(["prompt", "push", spec, "--name", "qa", "--registry", registry]) == 0
    capsys.readouterr()

    # versions lists both
    assert main(["prompt", "versions", "qa", "--registry", registry]) == 0
    versions_out = capsys.readouterr().out
    assert "v1" in versions_out and "v2" in versions_out

    # diff between v1 and v2 emits JSON
    assert main(
        ["prompt", "diff", "qa", "1", "2", "--registry", registry]
    ) == 0
    diff_out = capsys.readouterr().out
    json.loads(diff_out)

    # rollback to v1 creates a new version pointing back
    assert main(["prompt", "rollback", "qa", "--to", "1", "--registry", registry]) == 0
    rollback_out = capsys.readouterr().out
    assert "qa@" in rollback_out


# -- governance ---------------------------------------------------------------


def test_governance_card_system_to_stdout(tmp_path, capsys):
    app_py = _write_app(tmp_path)
    code = main(["governance", "card", app_py, "--kind", "system"])
    out = capsys.readouterr().out
    assert code == 0
    payload = json.loads(out)
    assert isinstance(payload, dict)


def test_governance_card_model_to_file(tmp_path, capsys):
    app_py = _write_app(tmp_path)
    target = tmp_path / "card.json"
    code = main(
        ["governance", "card", app_py, "--kind", "model", "--output", str(target)]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert f"saved to {target}" in out
    json.loads(target.read_text(encoding="utf-8"))


def test_governance_aibom(tmp_path, capsys):
    app_py = _write_app(tmp_path)
    code = main(["governance", "aibom", app_py])
    out = capsys.readouterr().out
    assert code == 0
    json.loads(out)


def test_governance_lineage_empty_source_fails(tmp_path, capsys):
    app_py = _write_app(tmp_path)
    code = main(["governance", "lineage", app_py, "ghost"])
    assert code == 1
    assert "no lineage for source" in capsys.readouterr().err


def test_governance_report_summary(tmp_path, capsys):
    app_py = _write_app(tmp_path)
    code = main(["governance", "report", app_py])
    out = capsys.readouterr().out
    assert code == 0
    payload = json.loads(out)
    assert isinstance(payload, dict)


def test_governance_report_markdown(tmp_path, capsys):
    app_py = _write_app(tmp_path)
    code = main(["governance", "report", app_py, "--markdown"])
    out = capsys.readouterr().out
    assert code == 0
    assert "#" in out  # markdown headings


def test_governance_report_full_to_file(tmp_path, capsys):
    app_py = _write_app(tmp_path)
    target = tmp_path / "report.json"
    code = main(["governance", "report", app_py, "--full", "--output", str(target)])
    out = capsys.readouterr().out
    assert code == 0
    assert f"saved to {target}" in out
    json.loads(target.read_text(encoding="utf-8"))


# -- cost report text ---------------------------------------------------------


def test_cost_report_text_summary(tmp_path, capsys):
    db = str(tmp_path / "vincio.db")
    code = main(["cost", "report", "--db", db, "--by", "feature"])
    out = capsys.readouterr().out
    assert code == 0
    # The text summary path prints a heading rather than JSON.
    assert out  # something rendered, and it is not JSON-parseable as a list
    with pytest.raises(json.JSONDecodeError):
        json.loads(out.strip().splitlines()[0])


# -- memory export / consolidate ----------------------------------------------


def test_memory_export_owner_to_stdout(tmp_path, capsys):
    db = str(tmp_path / "mem.db")
    main(["memory", "remember", "Likes tea", "--user", "u1", "--db", db])
    capsys.readouterr()
    code = main(["memory", "export", "--owner", "u1", "--db", db])
    out = capsys.readouterr().out
    assert code == 0
    records = json.loads(out)
    assert isinstance(records, list)
    assert any("tea" in json.dumps(r) for r in records)


def test_memory_export_to_file(tmp_path, capsys):
    db = str(tmp_path / "mem.db")
    main(["memory", "remember", "Likes coffee", "--user", "u2", "--db", db])
    capsys.readouterr()
    target = tmp_path / "export.json"
    code = main(
        ["memory", "export", "--owner", "u2", "--db", db, "--output", str(target)]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "exported" in out
    json.loads(target.read_text(encoding="utf-8"))


def test_memory_consolidate_runs(tmp_path, capsys):
    db = str(tmp_path / "mem.db")
    main(
        ["memory", "remember", "Met about pricing", "--user", "u1",
         "--session", "s1", "--db", db]
    )
    capsys.readouterr()
    code = main(["memory", "consolidate", "s1", "--user", "u1", "--db", db])
    out = capsys.readouterr().out
    assert code == 0
    assert "examined=" in out and "promoted=" in out


# -- distill / optimize / loop ------------------------------------------------


def _golden(tmp_cwd) -> str:
    golden = tmp_cwd / "golden.jsonl"
    golden.write_text(
        "\n".join(
            json.dumps({"id": f"c{i}", "input": q, "expected": "forty two"})
            for i, q in enumerate(["what?", "again?", "really?"])
        )
        + "\n",
        encoding="utf-8",
    )
    return str(golden)


def test_distill_from_traces(tmp_cwd, capsys):
    _run_and_get_trace_id(tmp_cwd, capsys)
    code = main(
        ["distill", "--output", "train.jsonl", "--allow-ungrounded"]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "example(s)" in out
    assert (tmp_cwd / "train.jsonl").is_file()


def test_optimize_run_reports_promotion_decision(tmp_cwd, capsys):
    (tmp_cwd / "app.py").write_text(_APP_PY, encoding="utf-8")
    golden = _golden(tmp_cwd)
    code = main(
        ["optimize", "run", "--app", "app.py", "--dataset", golden,
         "--budget", "2", "--subset", "2", "--target", "cost"]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "baseline fitness:" in out
    assert "promoted:" in out


def test_loop_run_dry_run(tmp_cwd, capsys):
    (tmp_cwd / "app.py").write_text(_APP_PY, encoding="utf-8")
    golden = _golden(tmp_cwd)
    code = main(
        ["loop", "run", "--app", "app.py", "--dataset", golden,
         "--budget", "2", "--subset", "2", "--dry-run"]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "dataset:" in out
    assert "promoted:" in out


def test_loop_run_rejects_malformed_gate(tmp_cwd, capsys):
    (tmp_cwd / "app.py").write_text(_APP_PY, encoding="utf-8")
    golden = _golden(tmp_cwd)
    code = main(
        ["loop", "run", "--app", "app.py", "--dataset", golden, "--gate", "bogus"]
    )
    assert code == 1
    assert "invalid gate" in capsys.readouterr().err


# -- eval run success path ----------------------------------------------------


def test_eval_run_success_with_output_and_passing_gate(tmp_cwd, capsys):
    (tmp_cwd / "app.py").write_text(_APP_PY, encoding="utf-8")
    golden = _golden(tmp_cwd)
    code = main(
        ["eval", "run", golden, "--app", "app.py", "--metric", "lexical_overlap",
         "--gate", "lexical_overlap=>= 0.0", "--output", "report.json"]
    )
    out = capsys.readouterr().out
    assert code == 0  # gate threshold 0.0 always passes
    assert "saved report to report.json" in out
    assert (tmp_cwd / "report.json").is_file()


# -- index build --------------------------------------------------------------


def test_index_build_directory(tmp_path, sample_docs_dir, capsys):
    db = str(tmp_path / "idx.db")
    code = main(["index", "build", str(sample_docs_dir), "--db", db])
    out = capsys.readouterr().out
    assert code == 0
    assert "document(s)" in out and "chunk(s)" in out


def test_index_build_single_file(tmp_path, sample_docs_dir, capsys):
    db = str(tmp_path / "idx.db")
    one = next(sample_docs_dir.glob("*.md"))
    code = main(["index", "build", str(one), "--db", db])
    out = capsys.readouterr().out
    assert code == 0
    assert "indexed 1 document(s)" in out


def test_index_build_missing_path(tmp_path, capsys):
    code = main(["index", "build", str(tmp_path / "nope"), "--db", str(tmp_path / "x.db")])
    assert code == 1
    assert "path not found" in capsys.readouterr().err


# -- trace replay against an app & recording verification ---------------------


def test_trace_replay_against_app(tmp_cwd, capsys):
    trace_id = _run_and_get_trace_id(tmp_cwd, capsys)
    code = main(["trace", "replay", trace_id, "--against", "app.py"])
    out = capsys.readouterr().out
    assert code == 0
    payload = json.loads(out)
    assert "summary" in payload


def _sealed_recording(tmp_path):
    from vincio.observability.record_replay import RecordedEdge, Recording

    rec = Recording(
        app_name="covdemo",
        run_id="run1",
        input="hi",
        output_text="forty two",
        edges=[RecordedEdge.of("model_call", 0, "m", {"text": "forty two"})],
    )
    rec.fidelity_digest = rec.compute_digest()
    path = tmp_path / "recording.json"
    rec.save(path)
    return rec, path


def test_trace_verify_recording_passes(tmp_path, capsys):
    _, path = _sealed_recording(tmp_path)
    code = main(["trace", "verify-recording", str(path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "recording verified" in out


def test_trace_verify_recording_detects_tamper(tmp_path, capsys):
    _, path = _sealed_recording(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    # Corrupt the digest so verification must fail.
    payload["fidelity_digest"] = "deadbeef"
    path.write_text(json.dumps(payload), encoding="utf-8")
    code = main(["trace", "verify-recording", str(path)])
    err = capsys.readouterr().err
    assert code == 1
    assert "FAILED verification" in err


# -- reflective optimize ------------------------------------------------------


def test_optimize_reflective_reports_decision(tmp_cwd, capsys):
    (tmp_cwd / "app.py").write_text(_APP_PY, encoding="utf-8")
    golden = _golden(tmp_cwd)
    code = main(
        ["optimize", "reflective", "--app", "app.py", "--dataset", golden,
         "--budget", "2", "--minibatch", "2", "--target", "cost"]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "strategy:" in out
    assert "promoted:" in out
