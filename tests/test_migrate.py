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
