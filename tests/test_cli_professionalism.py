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

# A synthetic deprecation map, so the scanner is exercised with a fixed shape
# independent of the aliases the library currently ships.
_FAKE = {
    "old_helper": Deprecation(
        name="old_helper", since="3.0", removed_in="4.0", alternative="new_helper"
    )
}

# The 7.5 factory-prefix normalization: every old factory name is a deprecated
# alias of its build_* replacement until 8.0 removes it.
_FACTORY_ALIASES = {
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


def test_collect_deprecations_reports_exactly_the_7_5_factory_aliases():
    # The only deprecated public API is the 7.5 make_*/create_* → build_*
    # factory-alias set; the collector must find each with its migration
    # metadata and nothing else.
    deprecations = collect_deprecations()
    assert set(deprecations) == set(_FACTORY_ALIASES)
    for old, new in _FACTORY_ALIASES.items():
        record = deprecations[old]
        assert record.since == "7.5"
        assert record.removed_in == "8.0"
        assert record.alternative == new


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


def test_scan_source_flags_submodule_and_aliased_module_access(tmp_path):
    # The forms the first 7.5 verification pass proved invisible: attribute
    # access through a dotted submodule, an aliased module import, and a
    # module bound by `from vincio import <submodule>`. All three must be
    # flagged, or `doctor` certifies a false clean after `migrate --write`.
    src = tmp_path / "pipeline.py"
    src.write_text(
        "import vincio.data\n"
        "import vincio.data as vd\n"
        "from vincio import data\n"
        "\n"
        "def go():\n"
        "    a = vincio.data.old_helper()\n"
        "    b = vd.old_helper()\n"
        "    c = data.old_helper()\n"
        "    return a, b, c\n",
        encoding="utf-8",
    )
    findings = scan_source(src, _FAKE)
    messages = [f.message for f in findings]
    assert sum("uses deprecated `vincio.data.old_helper`" in m for m in messages) == 3
    for finding in findings:
        assert finding.remediation == "use new_helper instead; removed in 4.0"


def test_scan_source_ignores_attribute_on_non_vincio_module(tmp_path):
    src = tmp_path / "other.py"
    src.write_text(
        "import numpy.data as vd\n\nx = vd.old_helper()\n", encoding="utf-8"
    )
    assert scan_source(src, _FAKE) == []


def test_scan_source_matches_bare_vincio_without_an_import(tmp_path):
    src = tmp_path / "reexport.py"
    src.write_text(
        "from myproject.compat import vincio\n\nx = vincio.old_helper()\n",
        encoding="utf-8",
    )
    findings = scan_source(src, _FAKE)
    assert any("uses deprecated `vincio.old_helper`" in f.message for f in findings)


def test_scan_source_ignores_relative_vincio_import(tmp_path):
    src = tmp_path / "vendored.py"
    src.write_text(
        "from .vincio import old_helper\n\nold_helper()\n", encoding="utf-8"
    )
    assert scan_source(src, _FAKE) == []


def test_scan_source_reports_multiline_attribute_at_the_token_line(tmp_path):
    src = tmp_path / "multiline.py"
    src.write_text(
        "import vincio\n"
        "x = (vincio.\n"
        "    old_helper)\n",
        encoding="utf-8",
    )
    findings = [f for f in scan_source(src, _FAKE) if "uses deprecated" in f.message]
    assert [f.line for f in findings] == [3]


def test_scan_source_flags_deprecated_keyword_on_vincio_calls(tmp_path):
    # The kwarg runway (verify_with= -> verifier=) is statically visible when
    # the call provably targets this library — a from-vincio name or a
    # vincio-module attribute. Receiver-typed method calls are runtime-covered.
    src = tmp_path / "kwargs.py"
    src.write_text(
        "import vincio\n"
        "from vincio import net_settlements\n"
        "\n"
        "ns1 = net_settlements([], verify_with=None)\n"
        "ns2 = vincio.net_settlements([], verify_with=None)\n"
        "other = dict(verify_with=1)  # not a vincio call\n",
        encoding="utf-8",
    )
    findings = [
        f for f in scan_source(src, collect_deprecations())
        if "verify_with=" in f.message
    ]
    assert [f.line for f in findings] == [4, 5]
    assert all("use verifier= instead; removed in 8.0" == f.remediation for f in findings)


def test_doctor_self_scan_stays_clean_with_kwarg_detection():
    # The library itself never passes the deprecated keyword.
    import vincio

    pkg_root = Path(vincio.__file__).resolve().parent
    report = run_doctor(pkg_root)
    assert report.ok, [f.message for f in report.findings]
