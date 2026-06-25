"""Learned skills: verified, content-addressed, versioned, composable procedures.

A :class:`LearnedSkill` is a named, typed, tool-using procedure the agent
*acquired itself* — distilled from a successful trajectory by ``app.cultivate``.
It carries a **precondition** (the state in which it applies), an ordered list of
**steps** (each a primitive :class:`~vincio.evals.environment.EnvAction` or an
invocation of an existing sub-skill, so skills *compose*), a **verifier** (the
end-state oracle it satisfies), and **provenance** (the cultivation cycle, the
reward it earned, and the no-regression verdict that promoted it).

This is deliberately distinct from the externally-authored ``SKILL.md``
procedural knowledge in :mod:`vincio.skills` (a :class:`~vincio.skills.Skill` is
*written by a human* and loaded as budgeted context). A learned skill is
content-addressed: its identity is the hash of its procedure, so two identical
procedures **deduplicate** and a changed procedure **versions** deterministically,
and :meth:`LearnedSkill.verify` recomputes that hash from the bytes alone so a
tampered procedure is caught offline. A :meth:`LearnedSkill.to_skill` projection
lets a learned skill be retrieved and cited through the very same
progressive-disclosure path as a written one.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from ..core.errors import CultivationError
from ..core.utils import stable_hash
from ..evals.environment import EnvAction, StateCheck

__all__ = [
    "SkillStep",
    "SkillProvenance",
    "LearnedSkill",
    "LearnedSkillLibrary",
]


_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return {t for t in _WORD.findall(text.lower()) if len(t) > 2}


class SkillStep(BaseModel):
    """One step of a learned procedure: a primitive action **or** a sub-skill call.

    Exactly one of :attr:`action` / :attr:`skill` is set. A ``skill`` step is how a
    learned skill *composes* an existing one — it is expanded into that skill's own
    (recursively composed) actions when the procedure is flattened for execution.
    """

    action: EnvAction | None = None
    skill: str | None = None

    def model_post_init(self, _context: Any) -> None:
        if (self.action is None) == (self.skill is None):
            raise CultivationError("a SkillStep sets exactly one of `action` or `skill`")


class SkillProvenance(BaseModel):
    """Where a learned skill came from, the audit trail of its acquisition."""

    cycle: str = ""
    objective: str = ""
    reward: float = 0.0
    attempts: int = 0
    verified: bool = False
    winning_steps: int = 0
    trajectory_hash: str = ""
    source_app: str = ""
    promotion: dict[str, Any] = Field(default_factory=dict)


class LearnedSkill(BaseModel):
    """A verified, content-addressed, versioned, composable learned procedure."""

    name: str
    version: int = 1
    description: str = ""
    objective: str = ""
    precondition: list[StateCheck] = Field(default_factory=list)
    steps: list[SkillStep] = Field(default_factory=list)
    verifier: list[StateCheck] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    provenance: SkillProvenance = Field(default_factory=SkillProvenance)
    skill_hash: str = ""

    # -- identity (content-addressing) --------------------------------------

    @property
    def requires(self) -> list[str]:
        """The sub-skills this procedure invokes, in first-use order (unique)."""
        seen: dict[str, None] = {}
        for step in self.steps:
            if step.skill is not None and step.skill not in seen:
                seen[step.skill] = None
        return list(seen)

    def body_facts(self) -> dict[str, Any]:
        """The canonical procedure used for the content hash.

        Version and provenance are **excluded** so that two identical procedures
        hash identically (and therefore deduplicate) regardless of when or by
        which cycle they were learned.
        """
        return {
            "objective": self.objective,
            "precondition": [c.model_dump(mode="json") for c in self.precondition],
            "steps": [s.model_dump(mode="json", exclude_none=True) for s in self.steps],
            "verifier": [c.model_dump(mode="json") for c in self.verifier],
        }

    def compute_hash(self) -> str:
        return stable_hash(self.body_facts(), length=32)

    def seal(self) -> LearnedSkill:
        """Stamp the content hash (idempotent)."""
        self.skill_hash = self.compute_hash()
        return self

    def verify(self) -> bool:
        """Recompute the content hash from the bytes — a tampered procedure fails."""
        return bool(self.skill_hash) and self.skill_hash == self.compute_hash()

    # -- retrieval & application --------------------------------------------

    def summary_line(self) -> str:
        """The always-disclosed index line (progressive-disclosure level 1)."""
        return (
            f"- {self.name} (v{self.version}): {self.description.strip() or self.objective.strip()}"
        )

    def match_score(self, query: str) -> float:
        """Deterministic relevance to *query* via token overlap. Range ``[0, 1]``."""
        haystack = _tokens(
            f"{self.name} {self.description} {self.objective} {' '.join(self.keywords)}"
        )
        needle = _tokens(query)
        if not haystack or not needle:
            return 0.0
        return len(haystack & needle) / len(haystack)

    def applies(self, state: dict[str, Any]) -> bool:
        """Whether this skill's precondition holds in *state* (empty ⇒ always)."""
        return all(check.evaluate(state).passed for check in self.precondition)

    def compose(
        self, library: LearnedSkillLibrary, *, _stack: tuple[str, ...] = ()
    ) -> list[EnvAction]:
        """Flatten this procedure into primitive actions, expanding sub-skills.

        Raises :class:`~vincio.core.errors.CultivationError` on a missing sub-skill
        or a composition cycle, so a composed skill is always executable or refused.
        """
        if self.name in _stack:
            cycle = " -> ".join((*_stack, self.name))
            raise CultivationError(f"skill composition cycle: {cycle}")
        actions: list[EnvAction] = []
        for step in self.steps:
            if step.action is not None:
                actions.append(step.action)
                continue
            sub = library.get(step.skill or "")
            if sub is None:
                raise CultivationError(
                    f"skill {self.name!r} requires missing sub-skill {step.skill!r}"
                )
            actions.extend(sub.compose(library, _stack=(*_stack, self.name)))
        return actions

    def to_skill(self) -> Any:
        """Project to a :class:`~vincio.skills.Skill` for budgeted, cited retrieval.

        Lets a learned procedure be surfaced through the same progressive-disclosure
        evidence path as a written ``SKILL.md`` skill, without coupling the learned
        library to the context compiler.
        """
        from ..skills import Skill

        lines = []
        for i, step in enumerate(self.steps, 1):
            if step.skill is not None:
                lines.append(f"{i}. apply skill `{step.skill}`")
            elif step.action is not None and step.action.kind == "tool":
                args = step.action.arguments or {}
                arg_text = f" {args}" if args else ""
                lines.append(f"{i}. call `{step.action.tool}`{arg_text}")
            elif step.action is not None:
                lines.append(f"{i}. {step.action.kind}: {step.action.text}".rstrip(": "))
        body = "\n".join(lines) or "(no steps)"
        instructions = (
            f"Objective: {self.objective}\n\nProcedure:\n{body}\n\n"
            f"Verified by {len(self.verifier)} end-state check(s)."
        )
        return Skill(
            name=self.name,
            description=self.description or self.objective,
            instructions=instructions,
            keywords=self.keywords,
            metadata={
                "origin": "learned_skill",
                "version": self.version,
                "skill_hash": self.skill_hash,
                "requires": self.requires,
            },
        )


class LearnedSkillLibrary:
    """A content-addressed library of learned skills with versioning and dedup.

    Skills are keyed by name with full version history; identity is the content
    hash of the procedure, so adding an identical procedure is a no-op
    (**dedup**) and adding a changed procedure under an existing name bumps the
    **version**. The library is itself content-bound — :attr:`library_hash` binds
    its current skills and :meth:`verify` recomputes it offline — and exposes the
    same ``relevant`` / ``evidence_for`` surface a :class:`~vincio.skills.SkillLibrary`
    does, so a learned skill is retrieved like memory and tools. A skill that
    stops paying its way is :meth:`demote`-d, never silently kept.
    """

    def __init__(self, skills: list[LearnedSkill] | None = None) -> None:
        self._versions: dict[str, list[LearnedSkill]] = {}
        self._current: dict[str, LearnedSkill] = {}
        self._by_hash: dict[str, LearnedSkill] = {}
        for skill in skills or []:
            self.add(skill)

    # -- membership ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self._current)

    def __contains__(self, name: str) -> bool:
        return name in self._current

    @property
    def skills(self) -> list[LearnedSkill]:
        """The current (latest-version, active) skills."""
        return list(self._current.values())

    @property
    def names(self) -> list[str]:
        return list(self._current)

    def get(self, name: str, *, version: int | None = None) -> LearnedSkill | None:
        if version is None:
            return self._current.get(name)
        for skill in self._versions.get(name, []):
            if skill.version == version:
                return skill
        return None

    def by_hash(self, skill_hash: str) -> LearnedSkill | None:
        return self._by_hash.get(skill_hash)

    def all_versions(self, name: str) -> list[LearnedSkill]:
        return list(self._versions.get(name, []))

    # -- mutation -----------------------------------------------------------

    def add(self, skill: LearnedSkill) -> LearnedSkill:
        """Add a skill, deduplicating by content hash and versioning by name.

        Returns the stored skill: the pre-existing one when the procedure is a
        byte-for-byte duplicate (dedup), or the freshly version-stamped skill
        otherwise. The skill is sealed first, so its content hash is authoritative.
        """
        skill.seal()
        existing = self._by_hash.get(skill.skill_hash)
        if existing is not None:
            return existing  # exact-duplicate procedure: dedup, never re-added
        history = self._versions.setdefault(skill.name, [])
        skill.version = max((s.version for s in history), default=0) + 1
        skill.seal()  # version is not in body_facts, hash is stable across the bump
        history.append(skill)
        self._current[skill.name] = skill
        self._by_hash[skill.skill_hash] = skill
        return skill

    def demote(self, name: str, *, reason: str = "") -> LearnedSkill | None:
        """Retire a skill from the active set (it no longer pays its way).

        Refused when another active skill still composes it — a depended-upon
        skill is load-bearing and is kept. Version history is preserved; only the
        active mapping is cleared, so the demotion is auditable and reversible.
        """
        skill = self._current.get(name)
        if skill is None:
            return None
        dependents = [
            s.name for s in self._current.values() if s.name != name and name in s.requires
        ]
        if dependents:
            raise CultivationError(f"cannot demote {name!r}: still required by {dependents}")
        del self._current[name]
        return skill

    def clone(self) -> LearnedSkillLibrary:
        """A shallow copy of the active set (for a trial promotion comparison)."""
        twin = LearnedSkillLibrary()
        twin._versions = {k: list(v) for k, v in self._versions.items()}
        twin._current = dict(self._current)
        twin._by_hash = dict(self._by_hash)
        return twin

    # -- composition --------------------------------------------------------

    def compose(self, name: str) -> list[EnvAction]:
        """The flattened primitive actions of *name*, expanding its sub-skills."""
        skill = self._current.get(name)
        if skill is None:
            raise CultivationError(f"no active skill named {name!r}")
        return skill.compose(self)

    # -- retrieval ----------------------------------------------------------

    def relevant(
        self, query: str, *, threshold: float = 0.05, limit: int = 3
    ) -> list[tuple[LearnedSkill, float]]:
        scored = [(s, s.match_score(query)) for s in self._current.values()]
        hits = [(s, score) for s, score in scored if score >= threshold]
        hits.sort(key=lambda pair: (pair[1], pair[0].name), reverse=True)
        return hits[:limit]

    def as_skill_library(self) -> Any:
        """Project the active skills into a :class:`~vincio.skills.SkillLibrary`."""
        from ..skills import SkillLibrary

        lib = SkillLibrary()
        for skill in self._current.values():
            lib.add(skill.to_skill())
        return lib

    def evidence_for(self, query: str, *, threshold: float = 0.05, limit: int = 3) -> list[Any]:
        """Index line (always) + relevant learned procedures (on match) as evidence."""
        items: list[Any] = self.as_skill_library().evidence_for(
            query, threshold=threshold, limit=limit
        )
        return items

    # -- content binding ----------------------------------------------------

    @property
    def library_hash(self) -> str:
        """A content hash binding the current skills (name, version, procedure)."""
        return stable_hash(
            sorted((s.name, s.version, s.skill_hash) for s in self._current.values()),
            length=32,
        )

    def verify(self) -> bool:
        """Every active skill recomputes its content hash — offline, from the bytes."""
        return all(skill.verify() for skill in self._current.values())

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "versions": {
                name: [s.model_dump(mode="json") for s in versions]
                for name, versions in self._versions.items()
            },
            "current": {name: s.version for name, s in self._current.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LearnedSkillLibrary:
        lib = cls()
        current = data.get("current", {})
        for name, versions in data.get("versions", {}).items():
            for raw in versions:
                skill = LearnedSkill.model_validate(raw)
                skill.seal()
                lib._versions.setdefault(name, []).append(skill)
                lib._by_hash[skill.skill_hash] = skill
                if current.get(name) == skill.version:
                    lib._current[name] = skill
        return lib
