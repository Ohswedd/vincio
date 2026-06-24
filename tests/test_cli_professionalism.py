"""CLI: `vincio doctor` and `vincio config migrate`, plus the doctor engine."""

from __future__ import annotations

from pathlib import Path

from vincio.cli.doctor import (
    Deprecation,
    collect_deprecations,
    run_doctor,
    scan_config,
    scan_source,
)
from vincio.cli.main import main

# A synthetic deprecation map, so the scanner is exercised without depending on
# the library actually shipping a deprecated symbol (it currently ships none).
_FAKE = {
    "old_helper": Deprecation(
        name="old_helper", since="3.0", removed_in="4.0", alternative="new_helper"
    )
}


def test_collect_deprecations_is_empty_on_a_clean_surface():
    # The public surface ships no deprecated APIs today; the collector must wire
    # up and return an empty map (not error).
    assert collect_deprecations() == {}


def test_vincio_package_is_doctor_clean_on_this_major():
    # The professionalism gate the 4.0 LTS major ships on: the library's own
    # source uses no deprecated public API and carries no config drift.
    import vincio

    pkg_root = Path(vincio.__file__).resolve().parent
    report = run_doctor(pkg_root)
    assert report.ok, [f.message for f in report.findings]


def test_scan_source_flags_deprecated_import_and_use(tmp_path):
    src = tmp_path / "app.py"
    src.write_text(
        "from vincio import old_helper\n"
        "import vincio\n"
        "\n"
        "def go():\n"
        "    old_helper()\n"
        "    return vincio.old_helper\n",
        encoding="utf-8",
    )
    findings = scan_source(src, _FAKE)
    messages = "\n".join(f.message for f in findings)
    assert "imports deprecated `old_helper`" in messages
    assert "uses deprecated `old_helper`" in messages
    assert "uses deprecated `vincio.old_helper`" in messages
    for finding in findings:
        assert finding.remediation == "use new_helper instead; removed in 4.0"


def test_scan_source_respects_import_alias(tmp_path):
    src = tmp_path / "aliased.py"
    src.write_text("from vincio import old_helper as oh\n\nx = oh()\n", encoding="utf-8")
    findings = scan_source(src, _FAKE)
    assert any("uses deprecated `old_helper`" in f.message for f in findings)


def test_scan_source_ignores_unrelated_names(tmp_path):
    src = tmp_path / "clean.py"
    src.write_text("from other import old_helper\nold_helper()\n", encoding="utf-8")
    # The name is not imported from vincio, so it is not flagged.
    assert scan_source(src, _FAKE) == []


def test_scan_config_flags_legacy_file(tmp_path):
    cfg = tmp_path / "vincio.yaml"
    cfg.write_text("project: old\nobservability:\n  exporter: console\n", encoding="utf-8")
    findings = scan_config(tmp_path)
    assert len(findings) == 1
    assert findings[0].kind == "config_drift"
    assert "vincio config migrate" in findings[0].remediation


def test_run_doctor_clean_project(tmp_path, capsys):
    (tmp_path / "app.py").write_text("import vincio\n", encoding="utf-8")
    assert main(["init", str(tmp_path), "--force"]) == 0
    capsys.readouterr()
    code = main(["doctor", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "no issues found" in out


def test_doctor_reports_issues_and_exits_nonzero(tmp_path, capsys):
    # legacy config => config drift finding, exit 1
    (tmp_path / "vincio.yaml").write_text(
        "project: old\nobservability:\n  exporter: console\n", encoding="utf-8"
    )
    report = run_doctor(tmp_path, deprecations=_FAKE)
    assert not report.ok
    assert any(f.kind == "config_drift" for f in report.findings)

    code = main(["doctor", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 1
    assert "config migrate" in out


def test_doctor_json_output(tmp_path, capsys):
    (tmp_path / "vincio.yaml").write_text(
        "project: old\nobservability:\n  exporter: console\n", encoding="utf-8"
    )
    code = main(["doctor", str(tmp_path), "--json"])
    out = capsys.readouterr().out
    import json

    payload = json.loads(out)
    assert payload["ok"] is False
    assert payload["findings"][0]["kind"] == "config_drift"
    assert code == 1


def test_cli_config_migrate_check_and_write(tmp_path, capsys):
    cfg = tmp_path / "vincio.yaml"
    cfg.write_text(
        "# yaml-language-server: $schema=./vincio.schema.json\n"
        "project: legacy\nobservability:\n  exporter: console\n",
        encoding="utf-8",
    )
    # --check: non-zero, file untouched
    assert main(["config", "migrate", str(cfg), "--check"]) == 1
    assert "console" in cfg.read_text(encoding="utf-8")
    capsys.readouterr()

    # write in place: upgrades, preserves the schema header
    assert main(["config", "migrate", str(cfg)]) == 0
    upgraded = cfg.read_text(encoding="utf-8")
    assert upgraded.startswith("# yaml-language-server:")
    assert "exporter: jsonl" in upgraded
    assert "schema_version:" in upgraded

    # re-run is a no-op
    capsys.readouterr()
    assert main(["config", "migrate", str(cfg)]) == 0
    assert "nothing to migrate" in capsys.readouterr().out


def test_cli_config_migrate_dry_run_does_not_write(tmp_path, capsys):
    cfg = tmp_path / "vincio.yaml"
    cfg.write_text("project: legacy\nobservability:\n  exporter: console\n", encoding="utf-8")
    assert main(["config", "migrate", str(cfg), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "exporter: jsonl" in out
    # disk unchanged
    assert "console" in cfg.read_text(encoding="utf-8")
