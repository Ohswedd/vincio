"""Vincio Agent Skills: portable ``SKILL.md`` procedural knowledge.

Load Anthropic-style skills and inject them through the context compiler with
progressive disclosure — budgeted, scored, and cited like any other context.

    from vincio.skills import load_skill
    app.add_skill("skills/pdf-processing")   # or load_skill(path) directly
"""

from __future__ import annotations

from .library import SkillLibrary
from .scripts import make_script_handler, register_skill_scripts
from .skill import (
    Skill,
    SkillError,
    SkillScript,
    load_skill,
    load_skills,
    parse_skill_md,
)

__all__ = [
    "Skill",
    "SkillScript",
    "SkillError",
    "SkillLibrary",
    "parse_skill_md",
    "load_skill",
    "load_skills",
    "make_script_handler",
    "register_skill_scripts",
]
