"""Config schema migrations: versioned, automatic, idempotent."""

from __future__ import annotations

from vincio.core.config import VincioConfig, load_config
from vincio.core.config_migrations import (
    CONFIG_SCHEMA_VERSION,
    detect_version,
    migrate,
    needs_migration,
)


def test_current_config_carries_schema_version():
    config = VincioConfig()
    assert config.schema_version == CONFIG_SCHEMA_VERSION


def test_detect_version_defaults_to_zero_for_legacy():
    assert detect_version({}) == 0
    assert detect_version({"schema_version": 1}) == 1
    assert detect_version({"schema_version": "nonsense"}) == 0


def test_legacy_config_needs_migration_and_is_stamped():
    legacy = {"project": "old"}
    assert needs_migration(legacy)
    result = migrate(legacy)
    assert result.from_version == 0
    assert result.to_version == CONFIG_SCHEMA_VERSION
    assert result.data["schema_version"] == CONFIG_SCHEMA_VERSION
    assert result.changed
    # input is not mutated
    assert "schema_version" not in legacy


def test_legacy_exporter_alias_is_canonicalized():
    result = migrate({"observability": {"exporter": "console"}})
    assert result.data["observability"]["exporter"] == "jsonl"
    assert any("console" in note for note in result.notes)


def test_current_config_is_a_noop():
    current = {"schema_version": CONFIG_SCHEMA_VERSION, "project": "x"}
    assert not needs_migration(current)
    result = migrate(current)
    assert not result.steps
    assert not result.changed
    assert result.data["schema_version"] == CONFIG_SCHEMA_VERSION


def test_migration_is_idempotent():
    once = migrate({"observability": {"exporter": "console"}}).data
    twice = migrate(once)
    assert not twice.steps
    assert twice.data == once


def test_load_config_auto_migrates_in_memory(tmp_path):
    path = tmp_path / "vincio.yaml"
    path.write_text("project: legacy\nobservability:\n  exporter: console\n", encoding="utf-8")
    config = load_config(path)
    # the stale file is upgraded in memory: version stamped, exporter canonical
    assert config.schema_version == CONFIG_SCHEMA_VERSION
    assert config.observability.exporter == "jsonl"
    # the file on disk is untouched until `vincio config migrate`
    assert "console" in path.read_text(encoding="utf-8")
