"""Sandboxed execution helpers (code execution sandbox).

`run_subprocess_sandboxed` executes a command in a separate process with a
timeout, output caps, a scrubbed environment, and an optional working
directory jail. `SandboxedPython` runs Python snippets in a subprocess with
``-I`` (isolated mode). These are OS-process isolation, not a security
boundary against a hostile kernel — appropriate for tool-grade isolation of
generated code; harden further with containers for adversarial workloads.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

from pydantic import BaseModel

from ..core.errors import ToolTimeoutError

__all__ = ["SandboxResult", "run_subprocess_sandboxed", "SandboxedPython"]

_SAFE_ENV_KEYS = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "TERM")


class SandboxResult(BaseModel):
    stdout: str
    stderr: str
    exit_code: int
    duration_ms: int
    truncated: bool = False


async def run_subprocess_sandboxed(
    command: list[str],
    *,
    timeout_s: float = 30.0,
    max_output_bytes: int = 1_000_000,
    cwd: str | Path | None = None,
    env_overrides: dict[str, str] | None = None,
    stdin_data: str | None = None,
) -> SandboxResult:
    """Run a command with a scrubbed environment and hard timeout."""
    import time

    env = {key: os.environ[key] for key in _SAFE_ENV_KEYS if key in os.environ}
    env.update(env_overrides or {})
    started = time.monotonic()
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.PIPE if stdin_data is not None else asyncio.subprocess.DEVNULL,
        cwd=str(cwd) if cwd else None,
        env=env,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(stdin_data.encode() if stdin_data is not None else None),
            timeout=timeout_s,
        )
    except TimeoutError as exc:
        process.kill()
        await process.wait()
        raise ToolTimeoutError(
            f"sandboxed command timed out after {timeout_s}s", tool=command[0]
        ) from exc
    truncated = len(stdout_bytes) > max_output_bytes or len(stderr_bytes) > max_output_bytes
    return SandboxResult(
        stdout=stdout_bytes[:max_output_bytes].decode("utf-8", errors="replace"),
        stderr=stderr_bytes[:max_output_bytes].decode("utf-8", errors="replace"),
        exit_code=process.returncode or 0,
        duration_ms=int((time.monotonic() - started) * 1000),
        truncated=truncated,
    )


class SandboxedPython:
    """Run Python snippets in an isolated subprocess (``python -I``).

    Isolated mode ignores environment variables and user site-packages; the
    snippet runs in a throwaway temp directory with no access to the parent
    process state.
    """

    def __init__(
        self,
        *,
        timeout_s: float = 15.0,
        max_output_bytes: int = 500_000,
        python_executable: str | None = None,
    ) -> None:
        self.timeout_s = timeout_s
        self.max_output_bytes = max_output_bytes
        self.python_executable = python_executable or sys.executable

    async def run(self, code: str) -> SandboxResult:
        with tempfile.TemporaryDirectory(prefix="vincio_sandbox_") as tmp:
            script = Path(tmp) / "snippet.py"
            script.write_text(code, encoding="utf-8")
            return await run_subprocess_sandboxed(
                [self.python_executable, "-I", str(script)],
                timeout_s=self.timeout_s,
                max_output_bytes=self.max_output_bytes,
                cwd=tmp,
            )
