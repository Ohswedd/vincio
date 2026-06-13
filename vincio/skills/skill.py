"""Agent Skills: portable ``SKILL.md`` procedural knowledge.

A skill is YAML frontmatter (at minimum ``name`` + ``description``) followed by
a Markdown body of instructions, optionally with bundled scripts. This is the
Anthropic Agent Skills format (donated to the Agentic AI Foundation). Vincio
loads them as *budgeted, scored, cited* context — not a privileged side
channel — with progressive disclosure: a one-line summary is always available,
and the full body is included only when the task is relevant (see
:class:`~vincio.skills.library.SkillLibrary`).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from ..core.errors import VincioError

__all__ = ["Skill", "SkillScript", "SkillError", "parse_skill_md", "load_skill", "load_skills"]


class SkillError(VincioError):
    """Raised when a SKILL.md file is missing or malformed."""


_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return {t for t in _WORD.findall(text.lower()) if len(t) > 2}


class SkillScript(BaseModel):
    """A script bundled with a skill, runnable as a sandboxed tool."""

    name: str
    path: str
    language: str = "python"  # python | shell
    description: str = ""


class Skill(BaseModel):
    """A parsed ``SKILL.md`` skill."""

    name: str
    description: str
    instructions: str = ""  # the Markdown body (progressive-disclosure level 2)
    keywords: list[str] = Field(default_factory=list)
    scripts: list[SkillScript] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    path: str | None = None

    def summary_line(self) -> str:
        """The always-disclosed index line (progressive-disclosure level 1)."""
        return f"- {self.name}: {self.description.strip()}"

    def match_score(self, query: str) -> float:
        """Deterministic relevance of this skill to *query* via token overlap of
        the name/description/keywords. Range [0, 1]."""
        haystack = _tokens(f"{self.name} {self.description} {' '.join(self.keywords)}")
        needle = _tokens(query)
        if not haystack or not needle:
            return 0.0
        overlap = haystack & needle
        return len(overlap) / len(haystack)


def parse_skill_md(text: str) -> tuple[dict[str, Any], str]:
    """Split a SKILL.md document into (frontmatter dict, Markdown body)."""
    stripped = text.lstrip("﻿")
    if not stripped.startswith("---"):
        raise SkillError("SKILL.md must begin with a YAML frontmatter block ('---')")
    # Frontmatter is delimited by the first two '---' lines.
    parts = re.split(r"^---\s*$", stripped, maxsplit=2, flags=re.MULTILINE)
    if len(parts) < 3:
        raise SkillError("SKILL.md frontmatter is not closed with a second '---'")
    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - defensive
        raise SkillError(f"invalid SKILL.md frontmatter: {exc}") from exc
    if not isinstance(meta, dict):
        raise SkillError("SKILL.md frontmatter must be a mapping")
    return meta, parts[2].strip()


def _keywords_from(meta: dict[str, Any]) -> list[str]:
    raw = meta.get("keywords") or meta.get("tags") or []
    if isinstance(raw, str):
        raw = [k.strip() for k in raw.split(",")]
    return [str(k) for k in raw if str(k).strip()]


def _discover_scripts(directory: Path, meta: dict[str, Any]) -> list[SkillScript]:
    scripts: list[SkillScript] = []
    seen: set[str] = set()
    # Explicit declarations in frontmatter take precedence.
    for entry in meta.get("scripts") or []:
        if isinstance(entry, str):
            entry = {"path": entry}
        path = str(entry.get("path", ""))
        if not path:
            continue
        name = str(entry.get("name") or Path(path).stem)
        lang = str(entry.get("language") or _language_for(path))
        scripts.append(SkillScript(name=name, path=path, language=lang, description=str(entry.get("description", ""))))
        seen.add(path)
    # Auto-discover a conventional scripts/ directory.
    scripts_dir = directory / "scripts"
    if scripts_dir.is_dir():
        for file in sorted(scripts_dir.iterdir()):
            if file.suffix not in (".py", ".sh"):
                continue
            rel = str(file.relative_to(directory))
            if rel in seen:
                continue
            scripts.append(
                SkillScript(name=file.stem, path=rel, language=_language_for(file.name))
            )
    return scripts


def _language_for(path: str) -> str:
    return "shell" if path.endswith(".sh") else "python"


def load_skill(path: str | Path) -> Skill:
    """Load one skill from a SKILL.md file or a directory containing one."""
    p = Path(path)
    if p.is_dir():
        md = p / "SKILL.md"
        directory = p
    else:
        md = p
        directory = p.parent
    if not md.is_file():
        raise SkillError(f"no SKILL.md found at {path}")
    meta, body = parse_skill_md(md.read_text(encoding="utf-8"))
    name = str(meta.get("name") or directory.name)
    description = str(meta.get("description") or "").strip()
    if not description:
        raise SkillError(f"skill {name!r} is missing a 'description' (required for disclosure)")
    extras = {k: v for k, v in meta.items() if k not in ("name", "description", "keywords", "tags", "scripts")}
    return Skill(
        name=name,
        description=description,
        instructions=body,
        keywords=_keywords_from(meta),
        scripts=_discover_scripts(directory, meta),
        metadata=extras,
        path=str(directory),
    )


def load_skills(path: str | Path) -> list[Skill]:
    """Load every skill under a directory (each skill is a subdir with SKILL.md)."""
    root = Path(path)
    if not root.is_dir():
        raise SkillError(f"not a directory: {path}")
    skills: list[Skill] = []
    if (root / "SKILL.md").is_file():
        return [load_skill(root)]
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "SKILL.md").is_file():
            skills.append(load_skill(child))
    return skills
