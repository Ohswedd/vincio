"""Prompt registry: versioned prompt store with tags, diffs, and rollback.

A local, file-backed registry (one JSON file per prompt name under
``.vincio/prompts``) — no hosted service. Every ``push`` of a changed spec
becomes a new immutable version keyed by ``spec_hash``; tags ("production",
"candidate") move between versions; ``rollback`` re-publishes an old version
as the new head; eval runs link to the exact version they measured.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..core.errors import PromptError
from ..core.utils import slugify, utcnow
from .optimizers import diff_rendered, diff_specs
from .templates import PromptSpec

__all__ = ["PromptVersion", "PromptRegistry"]


class PromptVersion(BaseModel):
    """One immutable version of a named prompt."""

    name: str
    version: int
    spec: PromptSpec
    spec_hash: str
    tags: list[str] = Field(default_factory=list)
    message: str = ""
    created_at: Any = Field(default_factory=utcnow)
    eval_runs: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def ref(self) -> str:
        return f"{self.name}@v{self.version}"


class PromptRegistry:
    """Versioned prompt store on the local filesystem.

    ``push`` is idempotent on content: re-pushing an unchanged spec returns
    the existing version instead of minting a new one. Tags are unique per
    prompt — tagging a version steals the tag from whichever version held it.
    """

    def __init__(self, directory: str | Path = ".vincio/prompts") -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    # -- storage ---------------------------------------------------------------

    def _path(self, name: str) -> Path:
        return self.directory / f"{slugify(name)}.json"

    def _load(self, name: str) -> list[PromptVersion]:
        path = self._path(name)
        if not path.is_file():
            return []
        records = json.loads(path.read_text(encoding="utf-8"))
        return [PromptVersion.model_validate(record) for record in records]

    def _save(self, name: str, versions: list[PromptVersion]) -> None:
        path = self._path(name)
        payload = [version.model_dump(mode="json") for version in versions]
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    # -- core API --------------------------------------------------------------

    def names(self) -> list[str]:
        return sorted(path.stem for path in self.directory.glob("*.json"))

    def versions(self, name: str) -> list[PromptVersion]:
        versions = self._load(name)
        if not versions:
            raise PromptError(f"prompt {name!r} not found in registry {self.directory}")
        return versions

    def push(
        self,
        spec: PromptSpec,
        *,
        name: str | None = None,
        tags: list[str] | None = None,
        message: str = "",
    ) -> PromptVersion:
        """Store a spec as a new version (or return the head if unchanged)."""
        name = name or spec.name
        versions = self._load(name)
        if versions and versions[-1].spec_hash == spec.spec_hash:
            version = versions[-1]
        else:
            version = PromptVersion(
                name=name,
                version=versions[-1].version + 1 if versions else 1,
                spec=spec,
                spec_hash=spec.spec_hash,
                message=message,
            )
            versions.append(version)
        if tags:
            self._apply_tags(versions, version.version, *tags)
        self._save(name, versions)
        return version

    def get(
        self, name: str, *, version: int | None = None, tag: str | None = None
    ) -> PromptVersion:
        """Resolve a version by number, tag, or head (latest)."""
        versions = self.versions(name)
        if version is not None:
            for item in versions:
                if item.version == version:
                    return item
            raise PromptError(f"prompt {name!r} has no version {version}")
        if tag is not None:
            for item in reversed(versions):
                if tag in item.tags:
                    return item
            raise PromptError(f"prompt {name!r} has no version tagged {tag!r}")
        return versions[-1]

    def spec(self, name: str, *, version: int | None = None, tag: str | None = None) -> PromptSpec:
        return self.get(name, version=version, tag=tag).spec

    # -- tags, rollback, diffs, eval links --------------------------------------

    @staticmethod
    def _apply_tags(versions: list[PromptVersion], version: int, *tags: str) -> PromptVersion:
        """Move tags onto one version of an in-memory history (no I/O)."""
        target: PromptVersion | None = None
        for item in versions:
            for label in tags:
                if label in item.tags and item.version != version:
                    item.tags.remove(label)
            if item.version == version:
                target = item
        if target is None:
            raise PromptError(f"no version {version} in history")
        for label in tags:
            if label not in target.tags:
                target.tags.append(label)
        return target

    def tag(self, name: str, version: int, *tags: str) -> PromptVersion:
        """Apply tags to a version; a tag lives on one version per prompt."""
        versions = self.versions(name)
        try:
            target = self._apply_tags(versions, version, *tags)
        except PromptError:
            raise PromptError(f"prompt {name!r} has no version {version}") from None
        self._save(name, versions)
        return target

    def rollback(self, name: str, *, to_version: int | None = None) -> PromptVersion:
        """Re-publish an earlier version as the new head (history is kept)."""
        versions = self.versions(name)
        if len(versions) < 2 and to_version is None:
            raise PromptError(f"prompt {name!r} has no earlier version to roll back to")
        target_number = to_version if to_version is not None else versions[-1].version - 1
        target = self.get(name, version=target_number)
        return self.push(
            target.spec.model_copy(deep=True),
            name=name,
            message=f"rollback to v{target.version}",
        )

    def diff(
        self, name: str, version_a: int, version_b: int, *, rendered: bool = False
    ) -> dict[str, Any]:
        """Field-level (and optionally rendered) diff between two versions."""
        a, b = self.get(name, version=version_a), self.get(name, version=version_b)
        result = diff_specs(a.spec, b.spec)
        result["version_a"], result["version_b"] = a.version, b.version
        if rendered:
            from .compiler import PromptCompiler

            compiler = PromptCompiler()
            text_a = "\n".join(m.text for m in compiler.compile(a.spec).messages)
            text_b = "\n".join(m.text for m in compiler.compile(b.spec).messages)
            result["rendered_diff"] = diff_rendered(text_a, text_b)
        return result

    def link_eval(
        self, name: str, version: int, report: Any, *, dataset: str | None = None
    ) -> PromptVersion:
        """Attach an eval run's summary to the version it measured.

        ``report`` is an :class:`vincio.evals.reports.EvalReport` (or a plain
        dict); only the aggregate summary is stored, keyed by report name.
        """
        versions = self.versions(name)
        target = next((item for item in versions if item.version == version), None)
        if target is None:
            raise PromptError(f"prompt {name!r} has no version {version}")
        if hasattr(report, "summary"):
            entry = {
                "report": getattr(report, "name", "eval"),
                "dataset": dataset or getattr(report, "dataset", ""),
                "created_at": str(getattr(report, "created_at", "")),
                "cases": len(getattr(report, "cases", [])),
                "metrics": {
                    metric: round(stats.get("mean", 0.0), 4)
                    for metric, stats in report.summary().items()
                },
                "gates": {
                    gate: value.get("passed") for gate, value in getattr(report, "gates", {}).items()
                },
            }
        else:
            entry = dict(report)
        target.eval_runs.append(entry)
        self._save(name, versions)
        return target
