"""Versioned, automatic ``vincio.yaml`` schema migrations.

A config file carries a ``schema_version``. When the schema evolves, a migration
upgrades older files mechanically instead of letting them silently drift: each
migration is a small, pure transform from version *N* to *N+1* that records the
concrete changes it made. :func:`migrate` chains every applicable migration in
order; :func:`load_config` applies them in memory on every load so a stale file
still validates against the current schema, and ``vincio config migrate``
persists the upgrade with a report of what changed.

Files written before versioning existed have no ``schema_version`` and are
treated as version ``0``. Adding a migration is additive: append a
:class:`Migration` and bump :data:`CONFIG_SCHEMA_VERSION`.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "Migration",
    "MigrationStep",
    "MigrationResult",
    "MIGRATIONS",
    "detect_version",
    "needs_migration",
    "migrate",
]

# The current config schema version. Bump by one whenever a migration is added.
CONFIG_SCHEMA_VERSION = 1

_VERSION_KEY = "schema_version"


@dataclass(frozen=True, slots=True)
class Migration:
    """A single ``from_version -> to_version`` config transform.

    ``apply`` mutates the (already-copied) config dict in place and returns a
    list of human-readable notes describing the concrete changes it made. An
    empty list means the migration only advanced the version stamp.
    """

    from_version: int
    to_version: int
    description: str
    apply: Callable[[dict[str, Any]], list[str]]


@dataclass(frozen=True, slots=True)
class MigrationStep:
    """One applied migration and the concrete changes it produced."""

    from_version: int
    to_version: int
    description: str
    notes: list[str]


@dataclass(slots=True)
class MigrationResult:
    """The outcome of migrating a config dict."""

    data: dict[str, Any]
    from_version: int
    to_version: int
    steps: list[MigrationStep] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        """True if any migration was applied (including a version stamp)."""
        return bool(self.steps)

    @property
    def notes(self) -> list[str]:
        """All concrete-change notes across applied steps, flattened."""
        out: list[str] = []
        for step in self.steps:
            out.extend(step.notes)
        return out


def _section(data: dict[str, Any], name: str) -> dict[str, Any] | None:
    value = data.get(name)
    return value if isinstance(value, dict) else None


# --- migration 0 -> 1 -------------------------------------------------------


def _v0_to_v1(data: dict[str, Any]) -> list[str]:
    """Introduce schema versioning and canonicalize the legacy exporter alias.

    Early Vincio docs listed ``observability.exporter: console`` as a value; the
    supported set is ``jsonl | memory | otel | none``. Map the legacy alias onto
    its modern equivalent so an old config keeps emitting traces.
    """
    notes: list[str] = []
    observability = _section(data, "observability")
    if observability is not None and observability.get("exporter") == "console":
        observability["exporter"] = "jsonl"
        notes.append("observability.exporter: 'console' -> 'jsonl' (legacy alias)")
    return notes


# Ordered, contiguous migrations. Each closes the gap to the next version.
MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        from_version=0,
        to_version=1,
        description="introduce schema versioning; canonicalize legacy exporter alias",
        apply=_v0_to_v1,
    ),
)


def detect_version(data: dict[str, Any]) -> int:
    """Return the declared ``schema_version`` (default ``0`` for legacy files)."""
    raw = data.get(_VERSION_KEY, 0)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def needs_migration(data: dict[str, Any]) -> bool:
    """True if *data* is behind :data:`CONFIG_SCHEMA_VERSION`."""
    return detect_version(data) < CONFIG_SCHEMA_VERSION


def migrate(data: dict[str, Any]) -> MigrationResult:
    """Chain every applicable migration, returning a fresh, upgraded dict.

    The input is not mutated. The result's ``schema_version`` is stamped to the
    current version even when no field changes were needed, so a re-run is a
    stable no-op.
    """
    working = copy.deepcopy(data)
    start = detect_version(working)
    steps: list[MigrationStep] = []
    for mig in MIGRATIONS:
        if mig.from_version < start or mig.from_version >= CONFIG_SCHEMA_VERSION:
            continue
        notes = mig.apply(working)
        working[_VERSION_KEY] = mig.to_version
        steps.append(
            MigrationStep(
                from_version=mig.from_version,
                to_version=mig.to_version,
                description=mig.description,
                notes=notes,
            )
        )
    if working.get(_VERSION_KEY) != CONFIG_SCHEMA_VERSION:
        working[_VERSION_KEY] = CONFIG_SCHEMA_VERSION
    return MigrationResult(
        data=working,
        from_version=start,
        to_version=CONFIG_SCHEMA_VERSION,
        steps=steps,
    )
