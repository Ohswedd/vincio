"""Run skill-bundled scripts as sandboxed tools.

Bundled scripts are not trusted code paths: they run through the same
resource-limited subprocess sandbox (timeout, output caps, scrubbed env, POSIX
``setrlimit``) as any other tool, and register through the permissioned tool
registry so RBAC/ABAC, approval, and the audit log all apply.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from ..stability import deprecated_alias
from ..tools.sandbox import run_subprocess_sandboxed
from .skill import Skill, SkillScript

__all__ = ["build_script_handler", "make_script_handler", "register_skill_scripts"]

_SCRIPT_SCHEMA = {
    "type": "object",
    "properties": {
        "args": {"type": "array", "items": {"type": "string"}},
        "stdin": {"type": "string"},
    },
}


def build_script_handler(skill: Skill, script: SkillScript):
    """Build an async handler that runs *script* in the subprocess sandbox."""
    script_path = str((Path(skill.path or ".") / script.path).resolve())

    async def handler(args: list[str] | None = None, stdin: str = "") -> dict[str, Any]:
        if script.language == "shell":
            command = ["sh", script_path, *(args or [])]
        else:
            command = [sys.executable, script_path, *(args or [])]
        result = await run_subprocess_sandboxed(
            command, stdin_data=stdin or None, timeout_s=15.0
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.exit_code,
            "truncated": result.truncated,
        }

    return handler


make_script_handler = deprecated_alias(
    build_script_handler,
    old_name="make_script_handler",
    since="7.5",
    removed_in="8.0",
)


def register_skill_scripts(
    registry: Any, skill: Skill, *, prefix: bool = True, permissions: list[str] | None = None
) -> list[str]:
    """Register every bundled script of *skill* as a sandboxed tool. Returns names.

    Scripts run as ``side_effects="external"`` tools — governed by the
    subprocess sandbox and the external-tool policy. Pass ``permissions`` (e.g.
    ``["skill:execute"]``) to additionally gate them behind an RBAC scope.
    """
    names: list[str] = []
    for script in skill.scripts:
        tool_name = f"{skill.name}.{script.name}" if prefix else script.name
        registry.register(
            build_script_handler(skill, script),
            name=tool_name,
            description=script.description
            or f"Bundled script {script.name!r} for skill {skill.name!r} (sandboxed).",
            input_schema=_SCRIPT_SCHEMA,
            permissions=permissions or [],
            side_effects="external",
        )
        names.append(tool_name)
    return names
