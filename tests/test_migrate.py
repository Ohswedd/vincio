"""CLI: `vincio migrate <target>` and the source-codemod engine."""

from __future__ import annotations

from vincio.cli.main import main
from vincio.cli.migrate import (
    RENAMES,
    SUPPORTED_TARGETS,
    SymbolRename,
    apply_rewrites,
    renames_for,
    run_migrate,
    scan_source,
)

# A synthetic rename table, so the codemod engine is exercised without the
# library actually renaming a public symbol (the 4.0 table ships empty).
_FAKE = {
    "old_name": SymbolRename(old="old_name", new="new_name", since="4.0", note="renamed"),
}


def test_four_zero_table_is_empty_clean_upgrade():
    # The additive-only 3.x contract held end to end: 4.0 renames nothing.
    assert "4.0" in SUPPORTED_TARGETS
    assert renames_for("4.0") == ()
    assert RENAMES["4.0"] == ()


# The 7.5 factory-prefix normalization, delivered mechanically at 8.0.
_EIGHT_ZERO_RENAMES = {
    "make_retail_environment": "build_retail_environment",
    "make_counter_environment": "build_counter_environment",
    "make_vault_environment": "build_vault_environment",
    "make_agent_solver": "build_agent_solver",
    "make_env_solver": "build_env_solver",
    "make_web_checkout": "build_web_checkout",
    "make_finetune_backend": "build_finetune_backend",
    "create_metadata_store": "build_metadata_store",
    "make_script_handler": "build_script_handler",
    "make_query_contract": "build_query_contract",
}


def test_eight_zero_table_ships_the_factory_renames():
    assert "8.0" in SUPPORTED_TARGETS
    table = renames_for("8.0")
    assert {r.old: r.new for r in table} == _EIGHT_ZERO_RENAMES
    for rename in table:
        assert rename.since == "7.5"
        assert rename.note == "factory-prefix normalization to build_*"
    # create_app (ASGI factory idiom) is deliberately exempt.
    assert "create_app" not in {r.old for r in table}


def test_eight_zero_round_trip_rewrites_import_and_use(tmp_path):
    src = tmp_path / "app.py"
    src.write_text(
        "from vincio.evals import make_agent_solver\n"
        "from vincio.evals.environment import make_vault_environment\n"
        "\n"
        "solver = make_agent_solver(lambda p: p)\n"
        "env = make_vault_environment()\n",
        encoding="utf-8",
    )
    report = run_migrate(tmp_path, target="8.0", write=True)
    assert not report.ok
    assert report.files_written == 1
    text = src.read_text(encoding="utf-8")
    assert "make_agent_solver" not in text
    assert "make_vault_environment" not in text
    assert "from vincio.evals import build_agent_solver" in text
    assert "solver = build_agent_solver(lambda p: p)" in text
    assert "env = build_vault_environment()" in text
    # a second pass finds nothing left to rewrite
    assert run_migrate(tmp_path, target="8.0").ok


def test_run_migrate_on_clean_table_reports_ok(tmp_path):
    (tmp_path / "app.py").write_text(
        "from vincio import ContextApp\napp = ContextApp()\n", encoding="utf-8"
    )
    report = run_migrate(tmp_path, target="4.0")
    assert report.ok
    assert report.rewrites == []
    assert report.files_scanned == 1


def test_run_migrate_unknown_target_raises():
    import pytest

    with pytest.raises(KeyError):
        run_migrate(".", target="9.9")


def test_scan_source_finds_import_use_and_attribute(tmp_path):
    src = tmp_path / "app.py"
    src.write_text(
        "from vincio import old_name\n"
        "import vincio\n"
        "\n"
        "def go():\n"
        "    old_name()\n"
        "    return vincio.old_name\n",
        encoding="utf-8",
    )
    rewrites = scan_source(src, _FAKE)
    # import token + bare use + attribute use = 3 rewrites, all old->new.
    assert len(rewrites) == 3
    assert all(rw.old == "old_name" and rw.new == "new_name" for rw in rewrites)


def test_scan_source_respects_import_alias(tmp_path):
    src = tmp_path / "aliased.py"
    src.write_text("from vincio import old_name as on\n\nx = on()\n", encoding="utf-8")
    rewrites = scan_source(src, _FAKE)
    # Only the imported token is rewritten; the local alias `on` is left alone.
    assert len(rewrites) == 1
    assert rewrites[0].line == 1


def test_scan_source_ignores_non_vincio_imports(tmp_path):
    src = tmp_path / "clean.py"
    src.write_text("from other import old_name\nold_name()\n", encoding="utf-8")
    assert scan_source(src, _FAKE) == []


def test_apply_rewrites_rewrites_exact_tokens(tmp_path):
    src = tmp_path / "app.py"
    source = (
        "from vincio import old_name\n"
        "import vincio\n"
        "\n"
        "def go():\n"
        "    old_name()\n"
        "    return vincio.old_name\n"
    )
    src.write_text(source, encoding="utf-8")
    rewrites = scan_source(src, _FAKE)
    updated = apply_rewrites(source, rewrites)
    assert "old_name" not in updated
    assert "from vincio import new_name" in updated
    assert "new_name()" in updated
    assert "vincio.new_name" in updated


def test_apply_rewrites_is_idempotent_and_position_safe(tmp_path):
    source = "from vincio import old_name\nold_name(); old_name()\n"
    src = tmp_path / "app.py"
    src.write_text(source, encoding="utf-8")
    rewrites = scan_source(src, _FAKE)
    once = apply_rewrites(source, rewrites)
    # Two uses on one line both rewrite; re-applying the same plan is a no-op
    # because the tokens no longer match `old`.
    assert once.count("new_name") == 3
    assert apply_rewrites(once, rewrites) == once


def test_run_migrate_write_applies_with_synthetic_table(tmp_path, monkeypatch):
    import vincio.cli.migrate as migrate_mod

    monkeypatch.setitem(migrate_mod.RENAMES, "test", (_FAKE["old_name"],))
    src = tmp_path / "app.py"
    src.write_text("from vincio import old_name\nold_name()\n", encoding="utf-8")

    # dry run leaves disk untouched
    report = run_migrate(tmp_path, target="test", write=False)
    assert not report.ok
    assert report.files_written == 0
    assert "old_name" in src.read_text(encoding="utf-8")

    # write applies in place
    report = run_migrate(tmp_path, target="test", write=True)
    assert report.files_written == 1
    text = src.read_text(encoding="utf-8")
    assert "old_name" not in text
    assert "new_name" in text


def test_cli_migrate_clean_tree_reports_no_changes(tmp_path, capsys):
    (tmp_path / "app.py").write_text("import vincio\n", encoding="utf-8")
    code = main(["migrate", "4.0", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "no source changes are required" in out


def test_cli_migrate_check_exit_code_with_synthetic_table(tmp_path, capsys, monkeypatch):
    import vincio.cli.migrate as migrate_mod

    # Register a synthetic target and expose it to the argparse choices.
    monkeypatch.setitem(migrate_mod.RENAMES, "8.0", (_FAKE["old_name"],))
    monkeypatch.setattr(migrate_mod, "SUPPORTED_TARGETS", tuple(migrate_mod.RENAMES))
    src = tmp_path / "app.py"
    src.write_text("from vincio import old_name\nold_name()\n", encoding="utf-8")

    code = main(["migrate", "8.0", str(tmp_path), "--check"])
    out = capsys.readouterr().out
    assert code == 1
    assert "migration available" in out
    # --check never writes
    assert "old_name" in src.read_text(encoding="utf-8")


def test_cli_migrate_json_output(tmp_path, capsys):
    (tmp_path / "app.py").write_text("import vincio\n", encoding="utf-8")
    code = main(["migrate", "4.0", str(tmp_path), "--json"])
    out = capsys.readouterr().out
    import json

    payload = json.loads(out)
    assert payload["target"] == "4.0"
    assert payload["ok"] is True
    assert payload["rewrites"] == []
    assert code == 0


def test_scan_source_rewrites_submodule_and_aliased_module_access(tmp_path):
    # The forms the first 7.5 verification pass proved invisible to the
    # codemod: attribute access through a dotted submodule, an aliased module
    # import, and a module bound by `from vincio import <submodule>`.
    src = tmp_path / "pipeline.py"
    src.write_text(
        "import vincio.data\n"
        "import vincio.data as vd\n"
        "from vincio import data\n"
        "\n"
        "def go():\n"
        "    a = vincio.data.old_name()\n"
        "    b = vd.old_name()\n"
        "    c = data.old_name()\n"
        "    return a, b, c\n",
        encoding="utf-8",
    )
    rewrites = scan_source(src, _FAKE)
    assert len(rewrites) == 3
    rewritten = apply_rewrites(src.read_text(encoding="utf-8"), rewrites)
    assert "vincio.data.new_name()" in rewritten
    assert "vd.new_name()" in rewritten
    assert "data.new_name()" in rewritten
    assert "old_name" not in rewritten
    # Idempotent: a second scan of the rewritten source finds nothing.
    src.write_text(rewritten, encoding="utf-8")
    assert scan_source(src, _FAKE) == []


def test_migrate_8_0_covers_submodule_attribute_form_end_to_end(tmp_path):
    # The exact false-clean scenario from the adversarial verification pass:
    # after `vincio migrate 8.0 --write`, no deprecated call may remain and
    # the doctor must agree.
    from vincio.cli.doctor import collect_deprecations
    from vincio.cli.doctor import scan_source as doctor_scan

    src = tmp_path / "pipeline.py"
    src.write_text(
        "from vincio.evals import make_retail_environment\n"
        "import vincio.data\n"
        "\n"
        "def main():\n"
        "    return (make_retail_environment('cancel_refund'),\n"
        "            vincio.data.make_query_contract(max_rows=100))\n",
        encoding="utf-8",
    )
    report = run_migrate(tmp_path, target="8.0", write=True)
    assert report.rewrites, "the 8.0 table must rewrite this file"
    rewritten = src.read_text(encoding="utf-8")
    assert "build_retail_environment" in rewritten
    assert "vincio.data.build_query_contract" in rewritten
    assert "make_" not in rewritten
    # And the doctor certifies a genuinely clean tree afterwards.
    assert doctor_scan(src, collect_deprecations()) == []
