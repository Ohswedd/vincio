"""0.9 CLI: init templates, config tooling, packs, and tui commands."""

from __future__ import annotations

import json

import pytest

from vincio.cli.main import main


@pytest.mark.parametrize("template", ["minimal", "rag", "agent", "eval"])
def test_init_templates_write_expected_files(tmp_path, template, capsys):
    root = tmp_path / template
    code = main(["init", str(root), "--template", template, "--project", "demo"])
    assert code == 0
    assert (root / "vincio.yaml").is_file()
    assert (root / "app.py").is_file()
    assert (root / "vincio.schema.json").is_file()
    # the config carries the editor schema hint and the chosen project name
    config_text = (root / "vincio.yaml").read_text()
    assert "yaml-language-server" in config_text
    assert "project: demo" in config_text
    # the schema file is valid JSON with the config title
    schema = json.loads((root / "vincio.schema.json").read_text())
    assert schema["title"] == "VincioConfig"
    if template == "rag":
        assert (root / "docs" / "welcome.md").is_file()
    if template == "eval":
        assert (root / "golden" / "eval.jsonl").is_file()


def test_init_refuses_existing_without_force(tmp_path):
    root = tmp_path / "proj"
    assert main(["init", str(root)]) == 0
    assert main(["init", str(root)]) == 1  # vincio.yaml exists
    assert main(["init", str(root), "--force"]) == 0


def test_init_with_provider(tmp_path):
    root = tmp_path / "p"
    main(["init", str(root), "--provider", "groq"])
    assert "default: groq" in (root / "vincio.yaml").read_text()


def test_config_schema_to_stdout_and_file(tmp_path, capsys):
    assert main(["config", "schema"]) == 0
    out = capsys.readouterr().out
    assert '"VincioConfig"' in out

    target = tmp_path / "schema.json"
    assert main(["config", "schema", "--output", str(target)]) == 0
    assert json.loads(target.read_text())["title"] == "VincioConfig"


def test_config_validate(tmp_path, capsys):
    root = tmp_path / "v"
    main(["init", str(root)])
    assert main(["config", "validate", str(root / "vincio.yaml")]) == 0
    assert "ok" in capsys.readouterr().out

    bad = tmp_path / "bad.yaml"
    bad.write_text("provider: {max_retries: not_a_number}\n")
    assert main(["config", "validate", str(bad)]) == 1


def test_config_show(tmp_path, capsys):
    root = tmp_path / "s"
    main(["init", str(root)])
    assert main(["config", "show", str(root / "vincio.yaml")]) == 0
    assert "project:" in capsys.readouterr().out


def test_packs_list_and_show(capsys):
    assert main(["packs", "list"]) == 0
    listing = capsys.readouterr().out
    assert "support" in listing and "engineering" in listing

    assert main(["packs", "show", "finance"]) == 0
    detail = capsys.readouterr().out
    assert "finance_metric" in detail
