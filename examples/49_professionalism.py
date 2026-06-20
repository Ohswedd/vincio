"""Professionalism & API ergonomics: a trustworthy surface.

Vincio's public surface is held to the same bar as its internals:

* every error carries a stable ``.code``, a ``.remediation`` hint, and a
  ``.docs_url`` deep link (the catalog is gated for completeness),
* ``vincio config migrate`` upgrades a ``vincio.yaml`` to the current schema,
  and stale files migrate in memory automatically on load,
* ``vincio doctor`` reports any deprecated API a project still uses (driven by
  the same ``stability_of`` metadata the library marks its own surface with),
* the whole package ships ``py.typed`` and a graduated ``mypy --strict`` set.

Everything here is offline and deterministic.
"""

from pathlib import Path

from vincio import VincioError, stability_of
from vincio.cli.doctor import run_doctor
from vincio.core.config import load_config
from vincio.core.config_migrations import migrate
from vincio.core.errors import ProviderAuthError


def show_actionable_errors() -> None:
    """Every error is catchable as one family and explains how to fix itself."""
    try:
        raise ProviderAuthError("401 from openai", provider="openai")
    except VincioError as exc:
        print(f"[{exc.code}] {exc.message}")
        print(f"  fix:  {exc.remediation}")
        print(f"  docs: {exc.docs_url}")


def migrate_a_legacy_config() -> None:
    """A pre-versioning config upgrades mechanically (here, an exporter alias)."""
    legacy = {"project": "demo", "observability": {"exporter": "console"}}
    result = migrate(legacy)
    print(f"config schema {result.from_version} -> {result.to_version}")
    for note in result.notes:
        print(f"  - {note}")


def auto_migrate_on_load(tmp: Path) -> None:
    """Stale files never silently drift: ``load_config`` migrates in memory."""
    path = tmp / "vincio.yaml"
    path.write_text("project: demo\nobservability:\n  exporter: console\n", encoding="utf-8")
    config = load_config(path)
    print(f"loaded exporter (canonicalized): {config.observability.exporter}")
    print(f"loaded schema_version: {config.schema_version}")


def doctor_a_project(tmp: Path) -> None:
    """The doctor flags deprecated APIs and config drift; here the tree is clean."""
    (tmp / "app.py").write_text("import vincio\nprint(vincio.__version__)\n", encoding="utf-8")
    (tmp / "vincio.yaml").write_text("schema_version: 1\nproject: demo\n", encoding="utf-8")
    report = run_doctor(tmp)
    print(
        f"doctor: scanned {report.files_scanned} file(s), "
        f"{report.deprecations_known} deprecated API(s) known, ok={report.ok}"
    )


def introspect_stability() -> None:
    """Tooling reads the same stability contract the library enforces."""
    record = stability_of(VincioError)
    print(f"VincioError stability level: {record['level']}")


if __name__ == "__main__":
    workdir = Path("project")
    workdir.mkdir(exist_ok=True)

    show_actionable_errors()
    print()
    migrate_a_legacy_config()
    print()
    auto_migrate_on_load(workdir)
    print()
    doctor_a_project(workdir)
    print()
    introspect_stability()
